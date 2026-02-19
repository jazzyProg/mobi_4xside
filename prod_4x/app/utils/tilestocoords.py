# tiler_convert.py
from __future__ import annotations
import os, glob
from pathlib import Path
from typing import Iterable, List, Tuple

from app.utils import filter as dedup

TILE_BASE  = 512
OVERLAY    = 160
TILE_FULL  = TILE_BASE + 2 * OVERLAY  # 832
IMAGESIZE = 8192

def tile_offset(row: int, col: int) -> Tuple[int, int]:
    """Смещение тайла (row, col) в глобальных координатах."""
    return col * TILE_BASE - OVERLAY, row * TILE_BASE - OVERLAY

def parse_tile_name(fname: str) -> Tuple[int, int]:
    """
    Из имени вида «xxx_5-13.txt» вытаскивает (row=5, col=13).
    Бросает ValueError, если формат не узнан.
    """
    name = Path(fname).stem
    row_col = name.rsplit('_', 1)[-1]
    row, col = map(int, row_col.split('-'))
    return row, col

def read_label_file(path: os.PathLike) -> Iterable[Tuple[int, List[float]]]:
    """
    Возвращает [(class_id, [x1, y1, x2, y2, ...]), ...]  из одного txt.
    """
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            yield int(parts[0]), list(map(float, parts[1:]))

def denormalize(coords: List[float], x_off: int, y_off: int) -> List[float]:
    """Переводит нормализованные координаты в глобальные пиксели без клипа."""
    out: List[float] = []
    for i in range(0, len(coords), 2):
        x = coords[i] * TILE_FULL + x_off
        y = coords[i + 1] * TILE_FULL + y_off
        out.extend([x, y])
    return out


def _filter_in_frame_pairs(transformed: List[float]) -> List[float]:
    filtered: List[float] = []
    max_coord = IMAGESIZE - 1.0
    for i in range(0, len(transformed), 2):
        x = transformed[i]
        y = transformed[i + 1]
        if 0.0 <= x <= max_coord and 0.0 <= y <= max_coord:
            filtered.extend([x, y])
    return filtered

def collect_labels(folder: os.PathLike) -> List[str]:
    """
    Читает все *.txt в `folder` и возвращает строки,
    готовые к записи в итоговый файл.
    """
    out_lines: List[str] = []
    for p in Path(folder).glob("*.txt"):
        try:
            row, col = parse_tile_name(p.name)
        except Exception:
            continue                      # skip «не наши» файлы
        x_off, y_off = tile_offset(row, col)
        for cls_id, coords in read_label_file(p):
            transformed = denormalize(coords, x_off, y_off)
            transformed = _filter_in_frame_pairs(transformed)
            if len(transformed) < 6:
                continue

            line = f"{cls_id} " + " ".join(map(str, transformed))
            line = dedup.remove_duplicate_points_in_line(line)
            if not line:
                continue
            out_lines.append(line)
    return out_lines

def save_labels(lines: Iterable[str], dst: os.PathLike) -> None:
    Path(dst).write_text("\n".join(lines))

# ----------------------------------------------------------------------
# CLI-оболочка
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser("Склейка координат из тайлов")
    ap.add_argument("-i", "--input",  required=True, help="Папка с *.txt")
    ap.add_argument("-o", "--output", default="all_coordinates.txt")
    ns = ap.parse_args()

    lines = collect_labels(ns.input)
    if not lines:
        print("Метки не найдены.")
        raise SystemExit(1)
    save_labels(lines, os.path.join(ns.input, ns.output))
    print("Готово:", ns.output)
