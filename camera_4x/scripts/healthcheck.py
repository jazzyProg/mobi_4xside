from __future__ import annotations

import os
import sys
import urllib.request

def main() -> int:
    host = os.getenv("HOST", "127.0.0.1")
    port = os.getenv("PORT", "8003")
    prefix = os.getenv("API_PREFIX", "").rstrip("/")
    url = f"http://{host}:{port}{prefix}/health"

    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return 0 if resp.status == 200 else 1
    except Exception:
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
