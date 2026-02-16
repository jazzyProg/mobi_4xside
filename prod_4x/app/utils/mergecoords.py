#!/usr/bin/env python3
# merge_coords.py – объединение и фильтрация контуров с удалением меньшей(ых) деталей
# ВАЖНО: после merge оставляет только крупнейший кластер класса 0 и игнорирует все
#        контуры классов 1 и 3 внутри меньших кластеров (не прямоугольник, а hull).

from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Tuple, Iterable, Dict
import numpy as np
import cv2
from collections import defaultdict
from scipy.spatial import cKDTree

Contour = List[Tuple[float, float]]

# ──────────────────────────────────────────────────────────────────────────
# Tokenizer / IO
# ──────────────────────────────────────────────────────────────────────────

def itertokens(lines: Iterable[str]):
    """Yield (class_id, [(x,y), ...]) from label lines."""
    for ln in lines:
        parts = ln.strip().split()
        if len(parts) < 3 or len(parts) % 2 != 1:
            continue
        try:
            cid = int(parts[0])
        except ValueError:
            continue
        if cid not in (0, 1, 3):
            continue
        coords = list(map(float, parts[1:]))
        if len(coords) < 6:
            continue
        pts = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        yield cid, pts

def read_contours_from_lines(lines: Iterable[str]):
    cls0: list[Contour] = []
    cls1: list[Contour] = []
    cls3: list[Contour] = []
    for cid, pts in itertokens(lines):
        if cid == 0:
            cls0.append(pts)
        elif cid == 1:
            cls1.append(pts)
        else:
            cls3.append(pts)
    return cls0, cls1, cls3

def read_contours(path: str | Path):
    return read_contours_from_lines(Path(path).read_text().splitlines())

def write_contour_line(out: list[str], cid: int, contour: Contour):
    if contour and len(contour) >= 3:
        flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in contour)
        out.append(f"{cid} {flat}")

def write_out(path: str | Path, cls0: Contour, cls1: list[Contour], cls3: list[Contour]):
    lines: list[str] = []
    write_contour_line(lines, 0, cls0)
    for c in cls1:
        write_contour_line(lines, 1, c)
    for c in cls3:
        # легкая очистка от подряд идущих дубликатов
        c_clean = remove_consecutive_duplicates(c, 0.1)
        write_contour_line(lines, 3, c_clean)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

# ──────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────

def remove_consecutive_duplicates(contour: Contour, thr: float = 0.1) -> Contour:
    if not contour:
        return []
    out: Contour = [contour[0]]
    thr2 = thr * thr
    for x, y in contour[1:]:
        if (x - out[-1][0]) ** 2 + (y - out[-1][1]) ** 2 > thr2:
            out.append((x, y))
    if len(out) > 1 and (out[0][0] - out[-1][0]) ** 2 + (out[0][1] - out[-1][1]) ** 2 < thr2:
        out.pop()
    return out

def centroid_np(pts: np.ndarray) -> tuple[float, float]:
    return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))

def _hole_center_and_radius(cnt: Contour) -> tuple[tuple[float, float], float]:
    """
    Устойчивый центр для отверстий (class 1):
    minEnclosingCircle лучше, чем mean(), когда контур "дугой" / обрезан / шумный.
    """
    pts = np.asarray(cnt, np.float32)
    if pts.shape[0] >= 3:
        (cx, cy), r = cv2.minEnclosingCircle(pts)
        return (float(cx), float(cy)), float(r)
    # fallback
    c = centroid_np(pts) if pts.size else (0.0, 0.0)
    return (float(c[0]), float(c[1])), 0.0

def hull_and_area(points_xy: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Return (hull_pts (N,2), area, hull_poly (N,1,2)) for cv2.pointPolygonTest."""
    if points_xy.shape[0] < 3:
        return points_xy.copy(), 0.0, points_xy.reshape(-1, 1, 2).astype(np.float32)
    hull = cv2.convexHull(points_xy.astype(np.float32))  # (N,1,2)
    area = float(cv2.contourArea(hull))
    hull_pts = hull.reshape(-1, 2)
    return hull_pts, area, hull

def cluster_labels_eps(coords: np.ndarray, eps: float) -> np.ndarray:
    """DBSCAN-like clustering with min_samples=1 via radius connectivity."""
    if len(coords) == 0:
        return np.empty((0,), dtype=int)
    tree = cKDTree(coords)
    neighbors = tree.query_ball_tree(tree, r=eps)
    labels = np.full(coords.shape[0], -1, int)
    cur = 0
    for i in range(coords.shape[0]):
        if labels[i] != -1:
            continue
        stack = [i]
        while stack:
            j = stack.pop()
            if labels[j] != -1:
                continue
            labels[j] = cur
            stack.extend(neighbors[j])
        cur += 1
    return labels

def dynamic_eps(centroids: np.ndarray,
                k: int = 2,
                mul: float = 2.5,
                clamp: tuple[float, float] = (700.0, 1800.0)) -> float:
    """Compute EPS from median 2nd-NN distance, then scale and clamp."""
    if len(centroids) < 2:
        return clamp[0]
    tree = cKDTree(centroids)
    dists, _ = tree.query(centroids, k=min(k, len(centroids)))
    # dists shape: (N,k); dists[:,0]=0.0 to self; take next column if exists
    base = np.median(dists[:, -1]) if dists.shape[1] >= 2 else np.median(dists[:, 0])
    val = float(base * mul)
    return float(np.clip(val, clamp[0], clamp[1]))

def combine_cluster_contours_np(cluster: list[Contour]) -> Contour:
    """Combine contours of the kept cluster into one ordered polygon."""
    pts = np.vstack(cluster, dtype=np.float32)
    if pts.shape[0] < 3:
        return []
    # Мягкое округление: шаг 0.5 px, затем удаление дубликатов
    rounded = np.round(pts * 2) / 2
    uniq, uniqidx = np.unique(rounded, axis=0, return_index=True)
    uniq = pts[np.sort(uniqidx)]
    if uniq.shape[0] < 3:
        return []
    # Упорядочивание по углу относительно центроида
    cx, cy = uniq.mean(0)
    ang = np.arctan2(uniq[:, 1] - cy, uniq[:, 0] - cx)
    order = np.argsort(ang)
    ordered = uniq[order]
    return [(float(x), float(y)) for x, y in ordered]

# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main(inputpath: str | Path,
         outputpath: str | Path | None = None,
         *,
         inputlines: Iterable[str] | None = None) -> tuple[Contour, list[Contour], list[Contour]] | None:
    """
    Алгоритм:
      1) читаем контуры классов 0/1/3
      2) кластеризуем слайсы класса 0 по центроидам (dynamic EPS)
      3) выбираем крупнейший кластер по площади hull (+ бонус за отверстия внутри)
      4) удаляем все остальные кластеры из класса 0
      5) фильтруем КЛАССЫ 1 и 3: удаляем контуры, центроиды которых попали внутрь hull-ов удалённых кластеров
      6) объединяем слайсы оставшегося кластера в один контур и записываем
    """
    if inputlines is None and inputpath is None:
        raise ValueError("Either inputpath or inputlines must be provided")

    if inputlines is None:
        inputlines = Path(inputpath).read_text().splitlines()

    # 1) read
    cls0raw, cls1raw, cls3raw = read_contours_from_lines(inputlines)

    print(f"[INFO] Найдено слайсов класса 0: {len(cls0raw)}")

    # 2) cluster C0 by centroids with dynamic EPS
    removed_hulls: list[np.ndarray] = []
    keep_indices: set[int] | None = None

    if len(cls0raw) >= 2:
        centroids = np.array([np.mean(contour, 0) for contour in cls0raw], dtype=np.float32)
        eps = dynamic_eps(centroids)
        labels = cluster_labels_eps(centroids, eps=eps)

        clusters: dict[int, list[int]] = defaultdict(list)
        for idx, lab in enumerate(labels):
            clusters[int(lab)].append(idx)

        print(f"[INFO] Найдено деталей (кластеров): {len(clusters)}")
        # 3) score clusters by hull-area + holes-inside
        stats: dict[int, Dict] = {}

        # FIX: центры отверстий считаем устойчиво, иначе при "дугах" дырки могут
        # переехать и испортить выбор лучшего кластера.
        cls1_centers = []
        for c in cls1raw:
            (cx, cy), _r = _hole_center_and_radius(c)
            cls1_centers.append((cx, cy))

        for cid, idxs in clusters.items():
            pts = np.vstack([np.asarray(cls0raw[i], np.float32) for i in idxs]) if idxs else np.empty((0, 2), np.float32)
            hull_pts, hull_area, hull_poly = hull_and_area(pts)
            holes_in = 0
            for cx, cy in cls1_centers:
                if cv2.pointPolygonTest(hull_poly, (float(cx), float(cy)), False) >= 0:
                    holes_in += 1
            # score: площадь + вес по количеству отверстий внутри
            score = hull_area + 5000.0 * holes_in
            stats[cid] = dict(idxs=idxs, hull=hull_poly, area=hull_area, holes=holes_in, score=score)
            print(f"[INFO]   Деталь {cid}: слайсов={len(idxs)}, hull_area={hull_area:.2f}, holes_in={holes_in}, score={score:.1f}")

        if len(stats) >= 2:
            best_id = max(stats, key=lambda k: stats[k]["score"])
            keep_indices = set(stats[best_id]["idxs"])
            # собрать hull-ы всех удаляемых кластеров
            for cid, st in stats.items():
                if cid == best_id:
                    continue
                removed_hulls.append(st["hull"])
            print(f"[INFO]   Выбрана деталь {best_id} (оставляем {len(keep_indices)} слайсов)")
        else:
            # один кластер — оставляем как есть
            only = next(iter(stats.values()))
            keep_indices = set(only["idxs"])
            print(f"[INFO]   Один кластер → оставляем все {len(keep_indices)} слайсов")

    # 4) remove other clusters from C0
    if keep_indices is not None:
        before = len(cls0raw)
        cls0raw = [contour for i, contour in enumerate(cls0raw) if i in keep_indices]
        print(f"[INFO] Удалены слайсы меньших деталей: {before - len(cls0raw)}; осталось {len(cls0raw)}")

    # 5) filter classes 1/3 by removed hulls (ignore anything inside smaller details)
    def is_inside_removed_hulls(cnt: Contour) -> bool:
        if not removed_hulls:
            return False

        # FIX: для class=1 снова используем устойчивый центр
        (cx, cy), _r = _hole_center_and_radius(cnt)

        for hull in removed_hulls:
            if cv2.pointPolygonTest(hull, (float(cx), float(cy)), False) >= 0:
                return True
        return False

    if removed_hulls:
        before1 = len(cls1raw)
        before3 = len(cls3raw)
        cls1raw = [c for c in cls1raw if not is_inside_removed_hulls(c)]
        cls3raw = [c for c in cls3raw if not is_inside_removed_hulls(c)]
        print(f"[INFO] Фильтрация классов 1/3 по меньшим деталям → 1: {before1}->{len(cls1raw)}, 3: {before3}->{len(cls3raw)}")

    # 6) combine kept cluster C0 into single contour
    cls0_final: Contour = []
    if cls0raw:
        cls0_final = combine_cluster_contours_np(cls0raw)

    # Class 1: легкая кластеризация по устойчивым центрам, объединение
    cls1_final: list[Contour] = []
    if cls1raw:
        centers_list = []
        for c in cls1raw:
            (cx, cy), _r = _hole_center_and_radius(c)
            centers_list.append((cx, cy))
        centers = np.array(centers_list, dtype=np.float32)

        # Оставляем вашу логику EPS, но центры теперь стабильнее
        eps1 = max(25.0, min(60.0, dynamic_eps(centers, mul=1.25, clamp=(20.0, 80.0))))

        labels1 = cluster_labels_eps(centers, eps=eps1)
        clusters1: dict[int, list[Contour]] = defaultdict(list)
        for idx, lab in enumerate(labels1):
            clusters1[int(lab)].append(cls1raw[idx])

        cls1_final = [combine_cluster_contours_np(cl) for cl in clusters1.values()]
        cls1_final = [c for c in cls1_final if len(c) >= 3]

    # Class 3: только фильтр по длине
    cls3_final = [c for c in cls3raw if len(c) >= 3]

    if outputpath is not None:
        write_out(outputpath, cls0_final, cls1_final, cls3_final)
        return None
    return cls0_final, cls1_final, cls3_final

# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python merge_coords.py <inputfile> <outputfile>")
        sys.exit(1)
    try:
        main(sys.argv[1], sys.argv[2])
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
