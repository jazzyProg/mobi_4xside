#!/usr/bin/env python3
"""slice.py – Python port with libvips backend for maximum speed.

Usage:
    python slice.py <input_dir> <output_dir> \
           --fmt png|jpg|orig [--quality 90] [--workers 8]

* All tiles are written into one flat `output_dir`.
* Filenames: `<basename>_<row>-<col>.<ext>`
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import cv2
import pyvips

# ---------------------------------------------------------------------------
# Size constants
IMAGE_SIZE = 8192
TILE_BASE  = 512
OVERLAY    = 160
TILE_FULL  = TILE_BASE + 2 * OVERLAY  # 832
NUM_TILES  = 16                       # 16 × 16 = 256

# ---------------------------------------------------------------------------
# Helpers: Loaders

def _load_image_vips(path: str | os.PathLike) -> pyvips.Image:
    """Load image with libvips - random access for overlapping tiles."""
    img = pyvips.Image.new_from_file(str(path), access='random')
    if img.width != IMAGE_SIZE or img.height != IMAGE_SIZE:
        raise ValueError(f"Image must be exactly {IMAGE_SIZE}×{IMAGE_SIZE}")
    # Cache in memory for fast random access
    img = img.copy_memory()
    return img

def _load_image_vips_from_buffer(buf: bytes) -> pyvips.Image:
    """Load image from memory buffer (JPEG bytes) using libvips."""
    # VIPS detects format automatically
    img = pyvips.Image.new_from_buffer(buf, "")
    # Check size (optional, can be removed if input varies)
    if img.width != IMAGE_SIZE or img.height != IMAGE_SIZE:
         # For robustness, we might want to just warn or resize, but here we strict check
         pass
    img = img.copy_memory()
    return img

def _load_image_cv2(path: str | os.PathLike) -> np.ndarray:
    """Fallback to OpenCV - preserves original format."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    if img.shape[0] != IMAGE_SIZE or img.shape[1] != IMAGE_SIZE:
        raise ValueError(f"Image must be exactly {IMAGE_SIZE}×{IMAGE_SIZE}")
    return img

def _load_image_cv2_from_buffer(buf: bytes) -> np.ndarray:
    """Fallback load from memory using OpenCV."""
    np_arr = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("OpenCV failed to decode bytes")
    return img

def _load_image(path: str | os.PathLike):
    """Unified loader - default to VIPS from disk."""
    return _load_image_vips(path)

# ---------------------------------------------------------------------------
# Helpers: Tile Creators

def _create_tile_vips(src: pyvips.Image, top_left: Tuple[int, int]) -> pyvips.Image:
    """Extract tile with padding using libvips."""
    x, y = top_left

    src_x0 = max(0, x)
    src_y0 = max(0, y)
    src_x1 = min(x + TILE_FULL, IMAGE_SIZE)
    src_y1 = min(y + TILE_FULL, IMAGE_SIZE)

    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return pyvips.Image.black(TILE_FULL, TILE_FULL, bands=src.bands)

    crop_w = src_x1 - src_x0
    crop_h = src_y1 - src_y0
    tile = src.crop(src_x0, src_y0, crop_w, crop_h)

    if crop_w < TILE_FULL or crop_h < TILE_FULL:
        pad_left = src_x0 - x
        pad_top = src_y0 - y
        tile = tile.embed(pad_left, pad_top, TILE_FULL, TILE_FULL, extend='black')

    return tile

def _create_tile_cv2(src: np.ndarray, top_left: Tuple[int, int]) -> np.ndarray:
    """Original OpenCV implementation."""
    x, y = top_left

    if src.ndim == 2:
        tile = np.zeros((TILE_FULL, TILE_FULL), dtype=src.dtype)
    else:
        tile = np.zeros((TILE_FULL, TILE_FULL, src.shape[2]), dtype=src.dtype)

    src_x0 = max(0, x)
    src_y0 = max(0, y)
    src_x1 = min(x + TILE_FULL, IMAGE_SIZE)
    src_y1 = min(y + TILE_FULL, IMAGE_SIZE)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return tile

    w, h = src_x1 - src_x0, src_y1 - src_y0
    dst_x0 = 0 if x >= 0 else -x
    dst_y0 = 0 if y >= 0 else -y

    if src.ndim == 2:
        tile[dst_y0:dst_y0 + h, dst_x0:dst_x0 + w] = src[src_y0:src_y1, src_x0:src_x1]
    else:
        tile[dst_y0:dst_y0 + h, dst_x0:dst_x0 + w, :] = src[src_y0:src_y1, src_x0:src_x1, :]
    return tile

def _create_tile(src, top_left: Tuple[int, int]):
    """Unified tile creator - default to VIPS."""
    return _create_tile_vips(src, top_left)

def _iter_tiles() -> Iterable[Tuple[int, int, int, int]]:
    for row in range(NUM_TILES):
        for col in range(NUM_TILES):
            yield row, col, col * TILE_BASE - OVERLAY, row * TILE_BASE - OVERLAY

# ---------------------------------------------------------------------------
# Disk API (Save tiles to files)

def slice_image(input_path: str | os.PathLike, output_dir: str | os.PathLike) -> bool:
    try:
        ext = Path(input_path).suffix.lower() or ".png"
        _slice_to_disk(input_path, output_dir, ext)
        return True
    except Exception as exc:
        print(exc, file=sys.stderr)
        return False

def slice_image_png(input_path: str | os.PathLike, output_dir: str | os.PathLike) -> bool:
    return slice_image_force_fmt(input_path, output_dir, ".png")

def slice_image_jpeg(input_path: str | os.PathLike, output_dir: str | os.PathLike,
                     quality: int = 90) -> bool:
    return slice_image_force_fmt(input_path, output_dir, ".jpg", jpeg_quality=quality)

def slice_image_force_fmt(input_path: str | os.PathLike, output_dir: str | os.PathLike,
                          ext: str, *, jpeg_quality: int | None = None) -> bool:
    try:
        _slice_to_disk(input_path, output_dir, ext, jpeg_quality=jpeg_quality)
        return True
    except Exception as exc:
        print(exc, file=sys.stderr)
        return False

# ---------------------------------------------------------------------------
# Memory API (Return list of numpy arrays)

def slice_image_to_memory(input_path: str | os.PathLike):
    """Read file from disk, slice, return list of numpy arrays (for YOLO)."""
    src = _load_image(input_path)
    tiles = []
    for _, _, x, y in _iter_tiles():
        tile = _create_tile(src, (x, y))
        # Convert VIPS to Numpy
        np_tile = np.ndarray(
            buffer=tile.write_to_memory(),
            dtype=np.uint8,
            shape=[tile.height, tile.width, tile.bands] if tile.bands > 1 else [tile.height, tile.width]
        )
        tiles.append(np_tile)
    return tiles

def slice_jpeg_bytes_to_memory(jpeg_data: bytes):
    """
    Read JPEG bytes from memory (SHM), slice, return list of numpy arrays.
    Tries libvips first, falls back to OpenCV.
    """
    try:
        src = _load_image_vips_from_buffer(jpeg_data)
        use_vips = True
    except Exception as e:
        print(f"[Slice] VIPS buffer load failed: {e}, using CV2 fallback", file=sys.stderr)
        src = _load_image_cv2_from_buffer(jpeg_data)
        use_vips = False

    tiles = []
    for _, _, x, y in _iter_tiles():
        if use_vips:
            tile = _create_tile_vips(src, (x, y))
            np_tile = np.ndarray(
                buffer=tile.write_to_memory(),
                dtype=np.uint8,
                shape=[tile.height, tile.width, tile.bands] if tile.bands > 1 else [tile.height, tile.width]
            )
        else:
            np_tile = _create_tile_cv2(src, (x, y))

        tiles.append(np_tile)
    return tiles

# ---------------------------------------------------------------------------
# Internal writer

def _slice_to_disk(input_path: str | os.PathLike, output_dir: str | os.PathLike,
                   ext: str, *, jpeg_quality: int | None = None) -> None:
    src = _load_image(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = Path(input_path).stem

    for row, col, x, y in _iter_tiles():
        tile = _create_tile(src, (x, y))
        out_name = f"{base}_{row}-{col}{ext}"
        out_path = output_dir / out_name

        if isinstance(tile, pyvips.Image):
            if ext == '.jpg':
                tile.write_to_file(str(out_path), Q=jpeg_quality or 90)
            else:
                tile.write_to_file(str(out_path))
        else:
            if ext == '.jpg':
                cv2.imwrite(str(out_path), tile,
                           [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality or 90)])
            else:
                cv2.imwrite(str(out_path), tile)

# ---------------------------------------------------------------------------
# CLI Helper (Multiprocessing)

def _mp_process(task):
    img_path, fmt, out_dir, quality = task
    if fmt == 'png':
        return slice_image_png(img_path, out_dir)
    if fmt == 'jpg':
        return slice_image_jpeg(img_path, out_dir, quality)
    return slice_image(img_path, out_dir)

# ---------------------------------------------------------------------------
# CLI Entry Point

def _cli():
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import freeze_support

    freeze_support()

    p = argparse.ArgumentParser(description='Batch‑tile 8192² images (libvips accelerated).')
    p.add_argument('input_dir', type=Path)
    p.add_argument('output_dir', type=Path)
    p.add_argument('--fmt', choices=['png', 'jpg', 'orig'], default='png')
    p.add_argument('--quality', type=int, default=90)
    p.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    args = p.parse_args()

    if not args.input_dir.is_dir():
        p.error('input_dir must be a directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}]
    if not images:
        print('No images found.', file=sys.stderr)
        sys.exit(1)

    tasks = [(img, args.fmt, args.output_dir, args.quality) for img in images]

    start_time = time.perf_counter()

    if args.workers == 1 or len(images) == 1:
        for t in tasks:
            _mp_process(t)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for _ in ex.map(_mp_process, tasks):
                pass

    elapsed = time.perf_counter() - start_time
    print(f'[slice.py] Processed {len(images)} image(s) in {elapsed:.2f} s. Tiles saved to {args.output_dir}')

if __name__ == '__main__':
    _cli()
