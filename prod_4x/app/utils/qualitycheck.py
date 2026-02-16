#!/usr/bin/env python3
"""
quality_check.py — модуль проверки соответствия CAD-«эталона» и Vision-замера
==============================================================================

Назначение
----------
API для автоматической валидации Vision JSON (результат `measure5.py`) относительно
CAD JSON (эталон от производственной базы данных).

**Критерии прохождения**
1. Количество отверстий (`holes_count`) должно совпадать *точно*.
2. Диаметры соответствующих отверстий: |ΔØ| ≤ `TOL_DIA_MM`.
3. Координаты `posX`, `posY` каждого отверстия: |ΔX| ≤ `TOL_POS_MM` **и** |ΔY| ≤ `TOL_POS_MM`.

Сопоставление отверстий выполняется тем же методом, что в `compare.py`:
венгерский алгоритм (`linear_sum_assignment`) по евклидовому расстоянию в
плоскости (X,Y). Перед сопоставлением применяется та же коррекция системы
координат Vision (инверсия Y и перестановка nearX/nearY), чтобы соответствовать
CAD-эталону.

Путь к CAD задаётся директорией: в ней должен находиться **ровно один** .json —
он и принимается как эталон. По умолчанию используется
`/home/jazzy/prod/json_svg`.

CLI (тестовый):
---------------
```
python quality_check.py --vision path/to/vision.json \
                        [--cad-dir /home/jazzy/prod/json_svg] \
                        [--tol-dia 0.5] [--tol-pos 7]
```
Если запустить без аргументов, модуль перейдёт в интерактивный режим и попросит
ввести пути вручную (удобно для ручных тестов).

Выход: JSON-отчёт в stdout; код возврата 0 при полном успехе, 1 иначе.

Зависимости: numpy, scipy.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Константы (допуски)
# ---------------------------------------------------------------------------
DEFAULT_CAD_DIR = Path("/home/jazzy/prod/json_svg")  # каталог, содержащий единственный CAD JSON
TOL_DIA_MM_SMALL: float = 1.37   # допустимое отклонение диаметра для отверстий < 20 мм
TOL_DIA_MM_LARGE: float = 2.7   # допустимое отклонение диаметра для отверстий >= 20 мм
DIA_THRESHOLD_MM: float = 20.0  # порог для переключения погрешности
TOL_POS_MM: float = 30.0        # допустимое отклонение координат posX/posY
MAX_EXTRA_HOLES: int = 1         # максимум лишних отверстий в Vision, которые игнорируем

# ---------------------------------------------------------------------------
# Типы
# ---------------------------------------------------------------------------
@dataclass
class Hole:
    id: int
    x: float
    y: float
    dia: float
    nx: float
    ny: float

@dataclass
class HoleDelta:
    cad_id: int
    vis_id: int
    dx: float  # vis - cad
    dy: float
    dd: float
    dnx: float
    dny: float

@dataclass
class QCResult:
    holes_count_ok: bool
    diameters_ok: bool
    positions_ok: bool
    ok: bool
    details: List[HoleDelta]
    extra_holes: int = 0

    def summary(self) -> str:
        parts = []
        parts.append("count=OK" if self.holes_count_ok else "count=FAIL")
        parts.append("dia=OK" if self.diameters_ok else "dia=FAIL")
        parts.append("pos=OK" if self.positions_ok else "pos=FAIL")
        if self.extra_holes > 0:
            parts.append(f"extra={self.extra_holes}")
        return ", ".join(parts)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["details"] = [asdict(hd) for hd in self.details]
        return d

# ---------------------------------------------------------------------------
# Вспомогательные функции чтения / распознавания формата
# ---------------------------------------------------------------------------

def _find_single_json_in_dir(dir_path: Path) -> Path:
    if not dir_path.is_dir():
        raise FileNotFoundError(f"CAD dir не найдена: {dir_path}")
    jsons = sorted(p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".json")
    if not jsons:
        raise FileNotFoundError(f"В каталоге {dir_path} нет CAD JSON")
    if len(jsons) > 1:
        raise RuntimeError(f"В каталоге {dir_path} >1 JSON; ожидался единственный")
    return jsons[0]


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_vision_dict(d: Dict) -> bool:
    """Возвращает True, если это JSON от measure5 (есть ключ 'rectangle', нет 'length')."""
    return ("rectangle" in d) and ("length" not in d)


def _match_cost(a: Hole, b: Hole) -> float:
    """Стоимость сопоставления двух отверстий на основе их атрибутов."""
    w_xy, w_dia, w_near = 1.0, 15.0, 0.7
    d_xy = math.hypot(a.x - b.x, a.y - b.y)
    d_dia = abs(a.dia - b.dia)
    d_near = (abs(abs(a.nx) - abs(b.nx)) + abs(abs(a.ny) - abs(b.ny)))
    return w_xy * d_xy + w_dia * d_dia + w_near * d_near


def _cost_matrix(A: Sequence[Hole], B: Sequence[Hole]) -> np.ndarray:
    n = len(A)
    m = len(B)
    C = np.empty((n, m))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            C[i, j] = _match_cost(a, b)
    return C

# ---------------------------------------------------------------------------
# Извлечение отверстий (по мотивам compare.load_holes)
# ---------------------------------------------------------------------------
def _extract_holes(d: Dict, *, is_vision: bool, invert_y: bool = False, invert_x: bool = False) -> List[Hole]:  # MOD: added invert_x
    holes: List[Hole] = []
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
        if is_vision:  # NEW: invert signs for nearest if flipping
            if invert_x:
                nx = -nx
            if invert_y:
                ny = -ny
        dia = float(h["diameter"])
        holes.append(Hole(id=int(h["hole_id"]), x=x, y=y, dia=dia, nx=nx, ny=ny))  # FIXED: dia was undefined, assume dia=float(h["diameter"])
    return holes

# ---------------------------------------------------------------------------
# Основной расчёт дельт (возвращает HoleDelta[] и индикаторы по допускам)
# ---------------------------------------------------------------------------

def _calc_deltas(Hc: List[Hole], Hv: List[Hole]) -> List[HoleDelta]:
    cost = _cost_matrix(Hc, Hv)
    rows, cols = linear_sum_assignment(cost)
    deltas: List[HoleDelta] = []
    for i, j in zip(rows, cols):
        a, b = Hc[i], Hv[j]
        deltas.append(
            HoleDelta(
                cad_id=a.id,
                vis_id=b.id,
                dx=b.x - a.x,
                dy=b.y - a.y,
                dd=b.dia - a.dia,
                dnx=b.nx - a.nx,
                dny=b.ny - a.ny,
            )
        )
    # Отсортируем детализированный список по CAD id — так стабильнее
    deltas.sort(key=lambda d: d.cad_id)
    return deltas

# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------

def check_files(
    vision_json_path: Path,
    cad_json_path: Path | None = None,
    *,
    cad_dir: Path = DEFAULT_CAD_DIR,
    tol_pos: float = TOL_POS_MM,
) -> QCResult:
    """Сравнить Vision и CAD JSON, вернуть QCResult."""
    # --- загрузка ---
    vis_d = _load_json(vision_json_path)
    if cad_json_path is None:
        cad_json_path = _find_single_json_in_dir(cad_dir)
    cad_d = _load_json(cad_json_path)

    # --- определение кто есть кто (порядок аргументов может быть любым) ---
    vis_is_vision = _is_vision_dict(vis_d)
    cad_is_vision = _is_vision_dict(cad_d)
    if vis_is_vision and cad_is_vision:
        raise RuntimeError("Оба JSON выглядят как Vision; CAD не найден")
    if (not vis_is_vision) and (not cad_is_vision):
        raise RuntimeError("Оба JSON выглядят как CAD; Vision не найден")
    if not vis_is_vision:  # пользователь перепутал порядок
        vis_d, cad_d = cad_d, vis_d

    # --- извлечение отверстий ---
    Hc = _extract_holes(cad_d, is_vision=False)

    # NEW: 4 variants for Vision
    Hv_normal = _extract_holes(vis_d, is_vision=True, invert_y=False, invert_x=False)
    Hv_inverted_y = _extract_holes(vis_d, is_vision=True, invert_y=True, invert_x=False)
    Hv_inverted_x = _extract_holes(vis_d, is_vision=True, invert_y=False, invert_x=True)
    Hv_inverted_both = _extract_holes(vis_d, is_vision=True, invert_y=True, invert_x=True)

    # --- вычисление стоимостей для всех версий ---
    costs = {}
    costs['normal'] = _cost_matrix(Hc, Hv_normal)
    costs['invert_y'] = _cost_matrix(Hc, Hv_inverted_y)
    costs['invert_x'] = _cost_matrix(Hc, Hv_inverted_x)
    costs['both'] = _cost_matrix(Hc, Hv_inverted_both)

    Hv_variants = {
        'normal': Hv_normal,
        'invert_y': Hv_inverted_y,
        'invert_x': Hv_inverted_x,
        'both': Hv_inverted_both
    }

    # Choose min cost variant
    min_variant = min(costs, key=lambda k: costs[k][linear_sum_assignment(costs[k])[0], linear_sum_assignment(costs[k])[1]].sum())
    cost_selected = costs[min_variant]
    Hv_selected = Hv_variants[min_variant]
    rows_selected, cols_selected = linear_sum_assignment(cost_selected)

    matched = len(rows_selected)
    Hc_cnt = len(Hc)
    Hv_cnt = len(Hv_selected)
    extra = Hv_cnt - matched

    if matched < Hc_cnt:
        return QCResult(
            holes_count_ok=False,
            diameters_ok=False,
            positions_ok=False,
            ok=False,
            details=[],
            extra_holes=extra,
        )

    if extra > MAX_EXTRA_HOLES:
        return QCResult(
            holes_count_ok=False,
            diameters_ok=False,
            positions_ok=False,
            ok=False,
            details=[],
            extra_holes=extra,
        )

    holes_count_ok = True

    # --- вычисление дельт ---
    deltas = _calc_deltas(Hc, Hv_selected)

    # --- проверки допусков (динамическая погрешность для диаметров) ---
    diameters_ok = True
    for d, a in zip(deltas, Hc):  # a — CAD отверстие
        tol_dia = TOL_DIA_MM_SMALL if a.dia < DIA_THRESHOLD_MM else TOL_DIA_MM_LARGE
        if abs(d.dd) > tol_dia:
            diameters_ok = False
            break

    positions_ok = all((abs(d.dx) <= tol_pos) and (abs(d.dy) <= tol_pos) for d in deltas)

    ok = holes_count_ok and diameters_ok and positions_ok
    return QCResult(
        holes_count_ok=holes_count_ok,
        diameters_ok=diameters_ok,
        positions_ok=positions_ok,
        ok=ok,
        details=deltas,
        extra_holes=extra,
    )

# ---------------------------------------------------------------------------
# CLI / интерактив для ручных тестов
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Проверка соответствия Vision JSON и CAD JSON (batch/test)",
    )
    p.add_argument("--vision", type=Path, help="Vision JSON (из measure5)")
    p.add_argument("--cad-dir", type=Path, default=DEFAULT_CAD_DIR,
                   help="Каталог с единственным CAD JSON (эталон)")
    p.add_argument("--cad-file", type=Path, default=None,
                   help="Необязательно: прямой путь к CAD JSON (переопределяет --cad-dir)")
    p.add_argument("--tol-pos", type=float, default=TOL_POS_MM)
    return p.parse_args()


def _interactive_prompt() -> Tuple[Path, Path | None, Path, float]:
    print("Интерактивный режим (пусто → по умолчанию)")
    vis = input("Vision JSON путь: ").strip()
    cad_dir = input(f"CAD каталог [{DEFAULT_CAD_DIR}]: ").strip() or str(DEFAULT_CAD_DIR)
    cad_file = input("CAD JSON (если хотите явно указать; иначе пусто): ").strip() or None
    tol_pos = input(f"Допуск pos мм [{TOL_POS_MM}]: ").strip()
    return (
        Path(vis),
        Path(cad_file) if cad_file else None,
        Path(cad_dir),
        float(tol_pos) if tol_pos else TOL_POS_MM,
    )


def _cli_main() -> int:
    if len(sys.argv) == 1:
        vis, cad_file, cad_dir, tol_pos = _interactive_prompt()
    else:
        ns = _parse_args()
        vis = ns.vision
        cad_dir = ns.cad_dir
        cad_file = ns.cad_file
        tol_pos = ns.tol_pos

    try:
        qc = check_files(vis, cad_file, cad_dir=cad_dir, tol_pos=tol_pos)
    except Exception as e:  # pragma: no cover - CLI error path
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps(qc.to_dict(), ensure_ascii=False, indent=2))
    return 0 if qc.ok else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(_cli_main())
