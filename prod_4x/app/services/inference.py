#!/usr/bin/env python3
"""inference.py — high‑throughput YOLO‑Seg inference (v3.2)

Key points
──────────
* **Singleton model** ‑ cached per weight file via `functools.lru_cache`  ➜ one copy
  in GPU memory for the whole process.
* **Thread‑safe GPU** – every forward pass is wrapped by a **global lock**.
  The lock object is taken from `main.INFERENCE_LOCK` if it exists; otherwise a
  private mutex is used.  This guarantees that only **one** tile batch touches
  the GPU at a time, no matter how many CPU threads are running.
* **Silent** – Ultralytics banner suppressed (`verbose=False`).
* **`release_gpu()`** – clears the LRU cache **and** empties CUDA memory;
  orchestrator calls it on graceful exit (Esc).
* CLI (`run_inference_on_folder`) stays fully compatible.
* NEW: `run_inference_on_tiles_batched` for processing tiles in batches.
"""
from __future__ import annotations

import argparse
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
from ultralytics import YOLO

# ───────── optional torch ─────────────────────────────────────────────────
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover – minimal env
    torch = None  # type: ignore

# ───────── global GPU mutex (shared with orchestrator) ────────────────────
import threading
try:
    # main.py sets INFERENCE_LOCK at import‑time
    from main import INFERENCE_LOCK  # type: ignore
except Exception:
    INFERENCE_LOCK: threading.Lock = threading.Lock()

# ───────── helpers ────────────────────────────────────────────────────────

def _mask_to_norm_contour(mask: np.ndarray, class_id: int) -> str:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return ""
    main = max(cnts, key=cv2.contourArea).squeeze(1).astype(np.float32)
    if main.shape[0] < 3:
        return ""
    h, w = mask.shape
    main[:, 0] /= w
    main[:, 1] /= h
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in main)
    return f"{class_id} {coords}"

# ───────── lazy model loader (LRU‑cached) ─────────────────────────────────

@lru_cache(maxsize=2)
def _get_model(
    model_path: str | os.PathLike,
    *,
    device: str | None,
    fp16: bool,
    verbose: bool,
):
    """Load YOLO once → share between threads."""
    mdl = YOLO(model_path, verbose=verbose)

    if device:
        mdl = mdl.to(device)

    # УБРАЛИ: if fp16: mdl = mdl.half() — это вызвало конфликт типов.
    # Теперь мы передаем управление точностью в момент вызова (predict).

    # light warm‑up (~3 ms) to avoid the very first 500 ms spike
    blank = np.zeros((832, 832, 3), np.uint8)

    with INFERENCE_LOCK:
        # ДОБАВИЛИ: half=fp16
        # Это прогреет модель в нужном режиме без перманентной поломки весов.
        mdl(blank, retina_masks=True, verbose=False, half=fp16)

    return mdl


def release_gpu() -> None:
    """Flush model cache **and** free CUDA memory (torch‑aware)."""
    _get_model.cache_clear()  # type: ignore[attr-defined]
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

# ───────── folder‑based (legacy) ─────────────────────────────────────────

def is_image_file(fname: str) -> bool:
    return fname.lower().endswith((
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))


def run_inference_on_folder(
    model_path: str,
    input_folder: str,
    output_folder: str,
    *,
    verbose: bool = False,
) -> None:
    os.makedirs(output_folder, exist_ok=True)
    model = _get_model(model_path, device=None, fp16=False, verbose=verbose)

    for fname in os.listdir(input_folder):
        if not is_image_file(fname):
            continue
        img_path = os.path.join(input_folder, fname)
        if verbose:
            print(f"[INFO] processing: {img_path}")
        with INFERENCE_LOCK:
            result = model(img_path, retina_masks=True, verbose=False)[0]
        _save_result_txt(result, Path(output_folder) / (Path(fname).stem + ".txt"))
        if verbose:
            print(f"[INFO] saved → {fname[:-4]}.txt")

# ───────── in‑memory tiling API ─────────────────────────────────────────–

def run_inference_on_tiles_seq(
    model_path: str | os.PathLike,
    tiles: Sequence[np.ndarray],
    *,
    out_dir: str | os.PathLike,
    stem: str,
    fp16: bool = False,
    device: str | None = None,
    verbose: bool = False,
) -> None:
    """Process 256 tiles **sequentially** under a global GPU lock."""
    NUM_TILES = 16
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _get_model(model_path, device=device, fp16=fp16, verbose=verbose)

    with INFERENCE_LOCK:  # exclusive GPU section
        for idx, tile in enumerate(tiles):
            # ensure 3‑channel BGR
            if tile.ndim == 2:
                tile = np.repeat(tile[:, :, None], 3, axis=2)
            elif tile.shape[2] == 4:
                tile = tile[:, :, :3]

            row, col = divmod(idx, NUM_TILES)
            txt_path = out_dir / f"{stem}_{row}-{col}.txt"
            result = model(tile, retina_masks=True, verbose=False)[0]
            _save_result_txt(result, txt_path)


def run_inference_on_tiles_batched(
    model_path: str | os.PathLike,
    tiles: Sequence[np.ndarray],
    *,
    out_dir: str | os.PathLike,
    stem: str,
    fp16: bool = False,
    device: str | None = None,
    verbose: bool = False,
    batch_size: int = 8, # NEW: размер батча
) -> None:
    """Process 256 tiles **in batches** under a global GPU lock."""
    NUM_TILES = 16
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _get_model(model_path, device=device, fp16=fp16, verbose=verbose)

    # Prepare tiles: ensure 3-channel BGR for all tiles
    prepared_tiles = []
    for tile in tiles:
        if tile.ndim == 2:
            tile = np.repeat(tile[:, :, None], 3, axis=2)
        elif tile.shape[2] == 4:
            tile = tile[:, :, :3]
        prepared_tiles.append(tile)

    with INFERENCE_LOCK:  # exclusive GPU section
        # Process tiles in batches
        for start_idx in range(0, len(prepared_tiles), batch_size):
            batch_of_tiles = prepared_tiles[start_idx : start_idx + batch_size]
            batch_indices = list(range(start_idx, min(start_idx + batch_size, len(prepared_tiles))))

            # Run inference on the batch
            results = model(batch_of_tiles, retina_masks=True, verbose=verbose)

            # Save results for each tile in the batch
            for idx_in_batch, (result, tile_idx) in enumerate(zip(results, batch_indices)):
                row, col = divmod(tile_idx, NUM_TILES)
                txt_path = out_dir / f"{stem}_{row}-{col}.txt"
                _save_result_txt(result, txt_path)


# ───────── TXT saver ─────────────────────────────────────────────────────

def _save_result_txt(result, txt_path: Path):
    lines: List[str] = []
    if result.masks is not None:
        h, w = result.orig_shape
        for i in range(len(result.masks)):
            mask = result.masks[i].data[0].cpu().numpy()
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            main = max(cnts, key=cv2.contourArea).squeeze(1).astype(np.float32)
            if main.shape[0] < 3:
                continue
            cls_id = int(result.boxes.cls[i].item())
            main[:, 0] /= w
            main[:, 1] /= h
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in main)
            lines.append(f"{cls_id} {coords}")
    txt_path.write_text("\n".join(lines), encoding="utf-8")

# ───────── CLI (folder mode) ─────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser("YOLO‑Seg inference (folder → TXT)")
    ap.add_argument("--model", "-m", required=True)
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--output", "-o", default="results")
    ap.add_argument("--verbose", action="store_true")
    # NEW: Add batch size argument for CLI if needed, though folder mode might not use it directly
    # ap.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    ns = ap.parse_args()

    run_inference_on_folder(ns.model, ns.input, ns.output, verbose=ns.verbose)
