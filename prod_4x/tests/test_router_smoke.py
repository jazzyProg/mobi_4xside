from __future__ import annotations


def test_app_import_and_routes_build():
    from app.main import create_app

    app = create_app()
    # ensure key endpoints exist
    paths = {r.path for r in app.router.routes}
    assert "/health" in paths
    assert "/status" in paths
