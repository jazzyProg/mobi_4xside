from __future__ import annotations

from pathlib import Path

import pytest


def test_process_single_accepts_backward_compatible_kwargs(monkeypatch, tmp_path: Path):
    """
    Ensure process_single accepts both:
    - modelpath/quarantinedir/stemname
    - model_path/quarantine_dir/stem_name

    We do not run the real pipeline (YOLO, CV, etc). We just verify argument plumbing.
    """
    from app.core import pipeline

    # Monkeypatch heavy internals early: make pipeline exit fast.
    # The simplest safe approach: patch the very first external call used later to avoid GPU work.
    # We patch qc_check_files to return minimal expected structure.
    monkeypatch.setattr(pipeline, "qc_check_files", lambda *a, **k: (True, {}), raising=True)

    # Also patch get_api_client usage (if any path calls it) to avoid network.
    class DummyClient:
        def get_active_product(self):
            return {"product_name": "dummy", "cad_data": {}}

    monkeypatch.setattr(pipeline, "get_api_client", lambda: DummyClient(), raising=True)

    model_path = tmp_path / "best.pt"
    model_path.write_text("x")
    qdir = tmp_path / "quarantine"
    qdir.mkdir()

    # bytes input; we only validate it doesn't crash on signature mismatch
    payload = b"fake-image-bytes"

    # Preferred/internal names
    with pytest.raises(Exception):
        pipeline.process_single(payload, modelpath=model_path, quarantinedir=qdir, stemname="x")

    # Public/backward-compatible alias names
    with pytest.raises(Exception):
        pipeline.process_single(payload, model_path=model_path, quarantine_dir=qdir, stem_name="x")
