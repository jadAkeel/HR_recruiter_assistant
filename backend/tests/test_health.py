from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import _resolve_keep_alive_url, _should_run_keep_alive
from app.main import create_app


def test_health() -> None:
    """
    Checks that health.
    """
    app = create_app()
    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_embedding_health() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.get("/api/v1/health/embeddings")

    assert response.status_code == 200
    body = response.json()
    assert body["provider"]
    assert "is_real" in body
    assert "auto_detect_lang" in body


def test_keep_alive_url_uses_render_hostname(monkeypatch) -> None:
    """
    Checks that Render deployments can derive the public health URL automatically.
    """
    monkeypatch.setattr(settings, "keep_alive_url", None)
    monkeypatch.setattr(settings, "api_prefix", "/api/v1")
    monkeypatch.setenv("RENDER_EXTERNAL_HOSTNAME", "demo.onrender.com")

    assert _resolve_keep_alive_url() == "https://demo.onrender.com/api/v1/health"


def test_keep_alive_url_prefers_explicit_setting(monkeypatch) -> None:
    """
    Checks that an explicit keep-alive URL overrides Render host detection.
    """
    monkeypatch.setattr(settings, "keep_alive_url", "https://example.com/ping")
    monkeypatch.setenv("RENDER_EXTERNAL_HOSTNAME", "demo.onrender.com")

    assert _resolve_keep_alive_url() == "https://example.com/ping"


def test_keep_alive_runs_on_render_even_without_explicit_setting(monkeypatch) -> None:
    """
    Checks that existing Render services do not need a blueprint env update.
    """
    monkeypatch.setattr(settings, "keep_alive_enabled", False)
    monkeypatch.setenv("RENDER_EXTERNAL_HOSTNAME", "demo.onrender.com")

    assert _should_run_keep_alive()
