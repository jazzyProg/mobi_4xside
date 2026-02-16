"""
Основной pipeline обработки изображений
Переделан из main.py с удалением TimeLogger и lampa.py
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import logging

# Импорты с учетом новой структуры и переименований
from app.utils import slice as slice_module
from app.services import inference
from app.utils import tilestocoords as t2c
from app.utils import mergecoords
from app.utils import filter as dedup
from app.services import measurement as measure5
from app.services import comparison as compare
from app.utils.qualitycheck import check_files as qc_check_files
from app.config import settings
from app.core.api_client import get_api_client

logger = logging.getLogger(__name__)


def slugify(txt: str, maxlen: int = 64) -> str:
    """Безопасное имя для папки"""
    out = [ch if ch.isalnum() or ch in '-_.' else '_' for ch in txt.strip()]
    slug = ''.join(out).strip('_') or 'detail'
    return slug[:maxlen]


def make_quarantine_dir(root: Path, cad_data: dict) -> Path:
    """Создать папку карантина на основе CAD данных"""
    product = slugify(str(cad_data.get('product_name', 'detail')))
    position = slugify(str(cad_data.get('position', '0')))
    now = datetime.now()
    datepart = now.strftime('%d%m%Y')
    timepart = now.strftime('%H%M%S')
    return root / f"{product}_{position}_{datepart}_{timepart}"


def deduplicate_file(path: Path, thickness: float):
    """Удалить дубликаты точек в файле"""
    raw = path.read_text(encoding='utf-8').splitlines()
    cleaned = []
    for ln in raw:
        ln = dedup.remove_duplicate_points_in_line(ln)
        if not ln:
            continue
        ln = dedup.keep_border_points_in_line(ln, thickness=thickness)
        if ln:
            cleaned.append(ln)
    path.write_text('\n'.join(cleaned), encoding='utf-8')


def is_ignorable_failure(qc: dict) -> bool:
    """Проверка на игнорируемую ошибку (из оригинального main.py)"""
    if not isinstance(qc, dict):
        return False

    IGNORABLE_ERROR_SUBSTRINGS = ["YOLO", "4"]

    err = str(qc.get('error', '')) or ''
    cmp = str(qc.get('compare_error', '')) or ''

    if any(s in err for s in IGNORABLE_ERROR_SUBSTRINGS):
        return True
    if 'No such file or directory' in cmp and '.json' in cmp:
        return True

    return False


def process_single(
    inputdata: Path | bytes,
    *,
    # Preferred names (internal)
    modelpath: Path | None = None,
    quarantinedir: Path | None = None,
    stemname: str | None = None,
    # Backward-compatible aliases (public API used by older routes)
    model_path: Path | None = None,
    quarantine_dir: Path | None = None,
    stem_name: str | None = None,
    dpi: int = settings.DPI,
    thickness: float = settings.THICKNESS_MM,
    session_id: str | None = None,
) -> tuple[bool, Path, dict]:
    """
    Основная функция обработки одного изображения.

    ИЗМЕНЕНИЯ по сравнению с оригинальным main.py:
    - Убрана зависимость от caddir - теперь получаем через API
    - Убран TimeLogger
    - Убран lampa.py
    - Используется api_client вместо modbus
    """

    # Resolve backward-compatible aliases
    if modelpath is None:
        modelpath = model_path
    if quarantinedir is None:
        quarantinedir = quarantine_dir
    if stemname is None:
        stemname = stem_name

    if modelpath is None or quarantinedir is None:
        raise ValueError("modelpath and quarantinedir are required")

    # 1. Определить stem для файлов
    is_bytes = isinstance(inputdata, bytes)
    if is_bytes:
        stem = stemname if stemname else f"frame_{int(time.time())}"
    else:
        stem = inputdata.stem

    logger.info(f"[{stem}] Starting pipeline processing")

    # Временная директория в RAM (/dev/shm)
    try:
        tmproot = Path(tempfile.mkdtemp(prefix='qc_pipe_', dir='/dev/shm'))
    except Exception:
        tmproot = Path(tempfile.mkdtemp(prefix='qc_pipe_'))

    # Пути для промежуточных файлов
    merged_path = tmproot / f"{stem}_merged.txt"
    annotated_jpg = tmproot / f"{stem}_annotated.jpg"
    vision_json = tmproot / f"{stem}.json"
    diff_txt = tmproot / f"{stem}_diff.txt"
    dev_plot = tmproot / f"{stem}_dev.jpg"
    diff_csv = tmproot / f"{stem}_diff.csv"

    qc_ok = True
    qc_dict = {"ok": True}
    error_msg: str | None = None

    # Получить API клиент
    api_client = get_api_client()
    cad_data: dict | None = None
    sourceimgpath: Path | None = None  # ДОБАВЛЕНО: инициализация
    cad_json_path: Path | None = None  # ДОБАВЛЕНО: инициализация

    try:
        try:
            # ===== ЭТАП 0: Получить активный продукт (CAD JSON) через API =====
            logger.info(f"[{stem}] Fetching active product via API")
            cad_data = api_client.get_active_product()

            # Сохранить во временную директорию для совместимости с qualitycheck
            cad_json_path = tmproot / "active_product.json"
            cad_json_path.write_text(
                json.dumps(cad_data, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )

            # ===== ЭТАП 1: Нарезка на тайлы =====
            logger.info(f"[{stem}] Slicing image to tiles")
            if is_bytes:
                # Из bytes (SHM от камеры)
                tiles = slice_module.slice_jpeg_bytes_to_memory(inputdata)
                sourceimgpath = tmproot / f"{stem}.jpg"
                sourceimgpath.write_bytes(inputdata)
            else:
                # Из файла
                tiles = slice_module.slice_image_to_memory(inputdata)
                sourceimgpath = inputdata

            # ===== ЭТАП 2: YOLO Inference =====
            logger.info(f"[{stem}] Running YOLO inference")
            infer_txt_dir = tmproot / "infer_txt"
            inference.run_inference_on_tiles_seq(
                model_path=str(modelpath),
                tiles=tiles,
                out_dir=infer_txt_dir,
                stem=stem,
                fp16=True,
                device='cuda',
                verbose=False
            )

            # ===== ЭТАП 3: Сбор координат =====
            logger.info(f"[{stem}] Collecting labels")
            lines = t2c.collect_labels(infer_txt_dir)
            if not lines:
                raise RuntimeError("YOLO не нашел объектов")

            all_coords = tmproot / "all_coords.txt"
            t2c.save_labels(lines, all_coords)

            # ===== ЭТАП 4: Объединение координат =====
            logger.info(f"[{stem}] Merging coordinates")
            mergecoords.main(str(all_coords), str(merged_path))

            # ===== ЭТАП 5: Дедупликация =====
            logger.info(f"[{stem}] Deduplicating points")
            deduplicate_file(merged_path, thickness)

            # ===== ЭТАП 6: Измерения =====
            logger.info(f"[{stem}] Running measurements")
            rect, holes, js_path = measure5.measure_board(
                merged_path,
                sourceimgpath,
                annotated_jpg
            )

            # js_path уже правильный (с расширением .json)
            vision_json = Path(js_path)

            logger.debug(f"[{stem}] Measurements complete: {len(holes)} holes found")

            # ===== ЭТАП 7: QC сравнение с CAD =====
            logger.info(f"[{stem}] Running QC check")
            qc_res = qc_check_files(
                vision_json,
                cad_json_path,
                cad_dir=None
            )
            qc_dict = qc_res.to_dict()
            qc_ok = bool(qc_res.ok)

            logger.info(f"[{stem}] QC result: {'PASS' if qc_ok else 'FAIL'}")

        except Exception as exc:
            qc_ok = False
            qc_dict = {"ok": False, "error": str(exc)}
            error_msg = str(exc)
            logger.error(f"[{stem}] Pipeline failed: {exc}", exc_info=True)

        # ===== ИСПРАВЛЕНО: Карантин при неудаче (всегда сохраняем) =====
        if not qc_ok and not is_ignorable_failure(qc_dict):
            logger.info(f"[{stem}] Saving to quarantine")

            try:
                # ИСПРАВЛЕНО: создаем карантинную папку даже без CAD данных
                if cad_data:
                    outdir = make_quarantine_dir(quarantinedir, cad_data)
                else:
                    # Fallback: общая папка для ошибок
                    outdir = Path(quarantinedir) / "errors" / stem

                outdir.mkdir(parents=True, exist_ok=True)

                # ИСПРАВЛЕНО: безопасно копируем исходное изображение
                if sourceimgpath and sourceimgpath.exists():
                    shutil.copy2(sourceimgpath, outdir / f"{stem}.jpg")
                elif is_bytes:
                    # Если файл еще не сохранен - сохраняем из bytes
                    (outdir / f"{stem}.jpg").write_bytes(inputdata)

                # Сравнение для анализа (только если есть CAD и vision)
                if cad_json_path and cad_json_path.exists() and vision_json.exists():
                    try:
                        compare.compare(
                            cad_json_path,
                            vision_json,
                            outdir / diff_txt.name,
                            outdir / dev_plot.name,
                            csv_file=outdir / diff_csv.name,
                            dpi=dpi
                        )
                    except Exception as exc:
                        qc_dict['compare_error'] = str(exc)
                        logger.warning(f"[{stem}] Compare failed: {exc}")

                # ИСПРАВЛЕНО: безопасно копируем промежуточные файлы
                for p in [annotated_jpg, merged_path, vision_json]:
                    if p.exists():
                        shutil.copy2(p, outdir / p.name)

                # Сохранить отчет
                (outdir / "qc_report.json").write_text(
                    json.dumps(qc_dict, ensure_ascii=False, indent=2),
                    encoding='utf-8'
                )

                logger.info(f"[{stem}] ✓ Saved to quarantine: {outdir}")

            except Exception as e:
                logger.error(f"[{stem}] Failed to save to quarantine: {e}", exc_info=True)

    finally:
        # Очистить tmproot
        shutil.rmtree(tmproot, ignore_errors=True)

    return qc_ok, vision_json, qc_dict
