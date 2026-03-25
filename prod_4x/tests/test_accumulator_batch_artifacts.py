from __future__ import annotations

import json
from pathlib import Path


def _reset_accum_state(accumulator_module):
    acc = accumulator_module.accumulator
    with acc.lock:
        for key in acc.counters:
            acc.counters[key] = 0
            acc.last_increment[key] = False
        acc.alarm_active = False
        acc.shift_key = ""
        for key in acc.sample_frames:
            acc.sample_frames[key] = []
            acc.artifact_saved_for_streak[key] = False


def test_save_single_accumulated_folder_with_five_images(monkeypatch, tmp_path: Path):
    from app.core import accumulator as accumulator_module

    _reset_accum_state(accumulator_module)
    monkeypatch.setattr(accumulator_module.settings, "BRAK_BASE_DIR", tmp_path, raising=False)

    class DummyClient:
        def get_active_product(self):
            return {"product_name": "B60"}

        def signal_alarm(self):
            return None

    monkeypatch.setattr(accumulator_module, "get_api_client", lambda: DummyClient(), raising=True)

    report = {"ok": False, "holes_ok": False}
    for idx in range(4):
        accumulator_module.increment_counters(report, frame_data=f"img{idx}".encode(), frame_id=idx)

    assert list(tmp_path.iterdir()) == []

    accumulator_module.increment_counters(report, frame_data=b"img5", frame_id=5)

    dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(dirs) == 1
    files = sorted([p.name for p in dirs[0].iterdir()])
    jpg_files = [name for name in files if name.endswith(".jpg")]
    assert len(jpg_files) == 5
    assert "defect_summary.json" in files

    payload = json.loads((dirs[0] / "defect_summary.json").read_text(encoding="utf-8"))
    assert payload["reason"] == "holes"
    assert payload["samples_count"] == 5
