from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional, Set


class UnsafePathError(ValueError):
    pass


def resolve_file_path(raw_path: str, *, base_dir: Path, allowed_suffixes: Set[str]) -> Path:
    """
    Resolve a path from DB and enforce sandboxing inside base_dir.
    """
    base = base_dir.expanduser().resolve()
    p = Path(raw_path)
    if not p.is_absolute():
        p = base / p
    rp = p.expanduser().resolve()

    try:
        rp.relative_to(base)
    except ValueError:
        raise UnsafePathError(f"path '{rp}' is outside of base_dir '{base}'")

    if rp.suffix.lower() not in allowed_suffixes:
        raise UnsafePathError(f"unexpected file suffix '{rp.suffix}' for '{rp.name}'")
    return rp


async def read_text_file(path: Path, *, encoding: str = "utf-8", max_bytes: int = 5_000_000) -> Optional[str]:
    """
    Read file content safely (async offload). Returns None if file doesn't exist.
    """
    if not path.exists():
        return None

    def _read() -> str:
        data = path.read_bytes()
        if len(data) > max_bytes:
            raise ValueError(f"file too large: {len(data)} bytes > {max_bytes}")
        return data.decode(encoding, errors="replace")

    return await asyncio.to_thread(_read)
