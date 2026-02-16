from __future__ import annotations

import json
import sys
import urllib.request


def main() -> int:
    url = "http://localhost:8001/health"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status != 200:
                return 1
            # Optional: validate response is JSON-ish (do not enforce schema)
            body = resp.read()
            if body:
                try:
                    json.loads(body.decode("utf-8"))
                except Exception:
                    # health endpoint might be plain text in future; treat as OK if 200
                    pass
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
