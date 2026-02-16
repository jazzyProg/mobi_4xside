import os
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from app.core.models import CapturedFrame

logger = logging.getLogger(__name__)

class DiskArchiver:
    def __init__(self, base_path: str, max_size_gb: int, auto_cleanup: bool = True):
        self.base_path = Path(base_path)
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.auto_cleanup = auto_cleanup

        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"DiskArchiver: {base_path}, max {max_size_gb}GB, auto_cleanup={auto_cleanup}")

    def save_frame(self, frame: CapturedFrame) -> str:
        dt = datetime.fromtimestamp(frame.metadata.timestamp)
        day_dir = self.base_path / dt.strftime("%Y-%m-%d") / dt.strftime("%H")
        day_dir.mkdir(parents=True, exist_ok=True)

        filename = f"frame_{frame.metadata.frame_id:08d}_{int(frame.metadata.timestamp * 1000)}.bin"
        filepath = day_dir / filename

        try:
            with open(filepath, 'wb') as f:
                f.write(frame.data)

            if self.auto_cleanup:
                self._cleanup_if_needed()

            logger.debug(f"Archived frame {frame.metadata.frame_id} to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save frame {frame.metadata.frame_id}: {e}")
            raise

    def load_frame(self, filepath: str) -> Optional[bytes]:
        try:
            with open(filepath, 'rb') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"Frame not found: {filepath}")
            return None
        except Exception as e:
            logger.error(f"Failed to load frame from {filepath}: {e}")
            return None

    def _cleanup_if_needed(self):
        total_size = self._get_total_size()

        if total_size <= self.max_size_bytes:
            return

        logger.warning(f"Disk usage {total_size/1024/1024/1024:.1f}GB exceeds limit, cleaning up...")

        day_dirs = []
        for item in self.base_path.rglob("*"):
            if item.is_dir() and item != self.base_path:
                day_dirs.append(item)

        day_dirs.sort(key=lambda x: x.stat().st_mtime)

        target_size = int(self.max_size_bytes * 0.8)

        for day_dir in day_dirs:
            current_size = self._get_total_size()
            if current_size <= target_size:
                break

            try:
                shutil.rmtree(day_dir)
                logger.info(f"Deleted old directory: {day_dir}")
            except Exception as e:
                logger.error(f"Failed to delete {day_dir}: {e}")

    def _get_total_size(self) -> int:
        total = 0
        for file in self.base_path.rglob("*.bin"):
            try:
                total += file.stat().st_size
            except Exception:
                pass
        return total

    def clear_session_files(self, session_id: Optional[str] = None):
        """
        Удалить файлы текущей или указанной сессии

        Args:
            session_id: ID сессии для удаления (если None - удаляет все)
        """
        deleted_count = 0
        deleted_size = 0

        try:
            if session_id:
                # Удалить файлы конкретной сессии
                pattern = f"*{session_id}*.bin"
                logger.info(f"Deleting files for session {session_id}...")
            else:
                # Удалить все файлы
                pattern = "*.bin"
                logger.info("Deleting all archived files...")

            for filepath in self.base_path.rglob(pattern):
                try:
                    file_size = filepath.stat().st_size
                    filepath.unlink()
                    deleted_count += 1
                    deleted_size += file_size
                except Exception as e:
                    logger.warning(f"Failed to delete {filepath}: {e}")

            logger.info(f"✓ Deleted {deleted_count} files ({deleted_size/1024/1024:.2f}MB)")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to clear session files: {e}")
            return 0

    def get_stats(self) -> dict:
        total_size = self._get_total_size()
        usage_percent = 0.0
        if self.max_size_bytes > 0:
            usage_percent = round(100 * total_size / self.max_size_bytes, 1)

        return {
            "total_size_gb": round(total_size / 1024 / 1024 / 1024, 2),
            "max_size_gb": round(self.max_size_bytes / 1024 / 1024 / 1024, 2),
            "usage_percent": usage_percent,
            "base_path": str(self.base_path)
        }

    def clear_all(self):
        return self.clear_session_files()
