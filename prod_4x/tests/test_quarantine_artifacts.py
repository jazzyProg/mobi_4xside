from __future__ import annotations

import json
from pathlib import Path


def test_quarantine_saves_only_required_artifacts(monkeypatch, tmp_path: Path):
    from app.core import pipeline
    monkeypatch.setattr(pipeline.settings, "DEBUG_MODE", True, raising=False)

    class DummyClient:
        def get_active_product(self):
            return {"product_name": "B60", "position": "56_1", "holes": []}

    class DummyQC:
        ok = False

        def to_dict(self):
            return {"ok": False, "error": "failed"}

    monkeypatch.setattr(pipeline, "get_api_client", lambda: DummyClient(), raising=True)
    monkeypatch.setattr(pipeline.slice_module, "slice_jpeg_bytes_to_memory", lambda _b: [b"tile"], raising=True)
    monkeypatch.setattr(pipeline.inference, "run_inference_on_tiles_seq", lambda **_k: None, raising=True)
    monkeypatch.setattr(pipeline.t2c, "collect_labels", lambda _p: ["1 2 3"], raising=True)
    monkeypatch.setattr(pipeline.t2c, "save_labels", lambda _lines, p: Path(p).write_text("1 2 3", encoding="utf-8"), raising=True)
    monkeypatch.setattr(pipeline.mergecoords, "main", lambda _src, dst: Path(dst).write_text("merged", encoding="utf-8"), raising=True)
    monkeypatch.setattr(pipeline, "deduplicate_file", lambda _p, _t: None, raising=True)

    def fake_measure(_merged, _src, annotated):
        Path(annotated).write_bytes(b"jpg")
        vis = Path(annotated).with_suffix(".json")
        vis.write_text(json.dumps({"rectangle": {"sides_mm": [1, 2, 1, 2]}, "holes": []}), encoding="utf-8")
        return {}, [], str(vis)

    monkeypatch.setattr(pipeline.measure5, "measure_board", fake_measure, raising=True)
    monkeypatch.setattr(pipeline, "qc_check_files", lambda *_a, **_k: DummyQC(), raising=True)

    def fake_compare(_cad, _vis, out_txt, out_plot, csv_file, dpi):
        Path(out_txt).write_text("diff", encoding="utf-8")
        Path(out_plot).write_bytes(b"plot")
        Path(csv_file).write_text("a,b\n1,2\n", encoding="utf-8")

    monkeypatch.setattr(pipeline.compare, "compare", fake_compare, raising=True)

    model = tmp_path / "best.pt"
    model.write_text("x", encoding="utf-8")
    qdir = tmp_path / "quarantine"

    ok, _vision_path, _qc = pipeline.process_single(
        b"fake-jpg",
        modelpath=model,
        quarantinedir=qdir,
        stemname="cam_1539",
    )

    assert ok is False
    saved_dirs = [p for p in qdir.iterdir() if p.is_dir()]
    assert len(saved_dirs) == 1
    files = {p.name for p in saved_dirs[0].iterdir()}

    assert "cam_1539.jpg" in files
    assert "cam_1539_annotated.jpg" in files
    assert "cam_1539_annotated.json" in files
    assert "cam_1539_diff.csv" in files
    assert "qc_report.json" in files

    assert "cam_1539_dev.jpg" not in files
    assert "cam_1539_diff.txt" not in files
    assert "cam_1539_merged.txt" not in files


def test_quarantine_not_saved_when_debug_disabled(monkeypatch, tmp_path: Path):
    from app.core import pipeline
    monkeypatch.setattr(pipeline.settings, "DEBUG_MODE", False, raising=False)

    class DummyClient:
        def get_active_product(self):
            return {"product_name": "B60", "position": "56_1", "holes": []}

    class DummyQC:
        ok = False

        def to_dict(self):
            return {"ok": False, "error": "failed"}

    monkeypatch.setattr(pipeline, "get_api_client", lambda: DummyClient(), raising=True)
    monkeypatch.setattr(pipeline.slice_module, "slice_jpeg_bytes_to_memory", lambda _b: [b"tile"], raising=True)
    monkeypatch.setattr(pipeline.inference, "run_inference_on_tiles_seq", lambda **_k: None, raising=True)
    monkeypatch.setattr(pipeline.t2c, "collect_labels", lambda _p: ["1 2 3"], raising=True)
    monkeypatch.setattr(pipeline.t2c, "save_labels", lambda _lines, p: Path(p).write_text("1 2 3", encoding="utf-8"), raising=True)
    monkeypatch.setattr(pipeline.mergecoords, "main", lambda _src, dst: Path(dst).write_text("merged", encoding="utf-8"), raising=True)
    monkeypatch.setattr(pipeline, "deduplicate_file", lambda _p, _t: None, raising=True)
    monkeypatch.setattr(pipeline.measure5, "measure_board", lambda *_a: ({}, [], str(tmp_path / "vision.json")), raising=True)
    monkeypatch.setattr(pipeline, "qc_check_files", lambda *_a, **_k: DummyQC(), raising=True)

    model = tmp_path / "best.pt"
    model.write_text("x", encoding="utf-8")
    qdir = tmp_path / "quarantine"

    ok, _vision_path, _qc = pipeline.process_single(
        b"fake-jpg",
        modelpath=model,
        quarantinedir=qdir,
        stemname="cam_1539",
    )

    assert ok is False
    assert not qdir.exists()


def test_make_quarantine_dir_uses_configured_timezone(monkeypatch, tmp_path: Path):
    from app.core import pipeline

    captured = {}

    class FakeNow:
        def strftime(self, fmt: str) -> str:
            return {"%d%m%Y": "24022026", "%H%M%S": "121512"}[fmt]

    class FakeDatetime:
        @staticmethod
        def now(tz):
            captured["tz"] = tz
            return FakeNow()

    monkeypatch.setattr(pipeline.settings, "TIMEZONE", "Europe/Moscow", raising=False)
    monkeypatch.setattr(pipeline, "datetime", FakeDatetime, raising=True)

    out = pipeline.make_quarantine_dir(tmp_path, {"product_name": "B60", "position": "56_1"})

    assert str(captured["tz"]) == "Europe/Moscow"
    assert out.name == "B60_56_1_24022026_121512"
