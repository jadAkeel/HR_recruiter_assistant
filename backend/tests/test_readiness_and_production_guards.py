import pytest
from fastapi.testclient import TestClient

from app.api import health as health_api
from app.core.config import Settings, settings
from app.main import create_app
from app.services import embedding as embedding_module
from app.services import task_queue


def test_ready_returns_503_when_dependency_is_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_collect_readiness() -> dict:
        return {"status": "degraded", "checks": {"redis": {"status": "degraded"}}}

    monkeypatch.setattr(health_api, "collect_readiness", fake_collect_readiness)

    client = TestClient(create_app())
    response = client.get("/api/v1/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_ready_returns_200_when_dependencies_are_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_collect_readiness() -> dict:
        return {"status": "ok", "checks": {"database": {"status": "ok"}}}

    monkeypatch.setattr(health_api, "collect_readiness", fake_collect_readiness)

    client = TestClient(create_app())
    response = client.get("/api/v1/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_production_cors_rejects_unconfigured_render_origins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "cors_origins_str", "https://trusted-frontend.onrender.com")
    client = TestClient(create_app())
    headers = {
        "Origin": "https://attacker.onrender.com",
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "authorization",
    }

    response = client.options("/api/v1/candidates", headers=headers)

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_production_cors_allows_configured_frontend(monkeypatch: pytest.MonkeyPatch) -> None:
    origin = "https://trusted-frontend.onrender.com"
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "cors_origins_str", origin)
    client = TestClient(create_app())
    headers = {
        "Origin": origin,
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "authorization",
    }

    response = client.options("/api/v1/candidates", headers=headers)

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_production_rejects_unowned_resource_compatibility() -> None:
    production_settings = Settings(
        environment="production",
        database_url="postgresql+asyncpg://db.example/test",
        jwt_secret_key="a-strong-production-secret-that-is-long-enough",
        cors_origins_str="https://frontend.example",
        trusted_hosts_str="api.example",
        allow_unowned_resources=True,
    )

    with pytest.raises(RuntimeError, match="ALLOW_UNOWNED_RESOURCES"):
        production_settings.validate_runtime()


@pytest.mark.asyncio
async def test_cv_queue_requires_redis_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    async def unavailable_redis():
        return None

    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(task_queue, "get_redis", unavailable_redis)

    with pytest.raises(RuntimeError, match="Redis is required"):
        await task_queue.enqueue_cv_processing(
            "Python",
            "candidate.txt",
            created_by_user_id="owner-id",
        )


@pytest.mark.asyncio
async def test_cv_queue_requires_owner_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "environment", "production")

    with pytest.raises(RuntimeError, match="require an owner"):
        await task_queue.enqueue_cv_processing("Python", "candidate.txt")


def test_production_ollama_embedding_constructs_without_hash_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "embedding_provider", "ollama")
    monkeypatch.setattr(settings, "ollama_embedding_model", "nomic-embed-text")
    monkeypatch.setattr(settings, "embedding_dimension", 768)

    embedding_module._get_cached_embedding_service.cache_clear()
    service = embedding_module.get_standard_embedding_service()
    embedding_module._get_cached_embedding_service.cache_clear()

    assert service is not None


def _preflight(client: TestClient, origin: str):
    return client.options(
        "/api/v1/candidates",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )


@pytest.mark.parametrize(
    "origin",
    [
        "https://x.onrender.com",
        "https://attacker.ngrok-free.app",
        "https://abcd.lhr.life",
        "http://localhost.attacker.com",
    ],
)
def test_development_cors_rejects_foreign_origins_by_default(monkeypatch: pytest.MonkeyPatch, origin: str) -> None:
    monkeypatch.setattr(settings, "environment", "development")
    monkeypatch.setattr(settings, "cors_allow_tunnels", False)
    client = TestClient(create_app())

    response = _preflight(client, origin)

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize("origin", ["http://localhost:4321", "http://127.0.0.1:8080", "http://localhost"])
def test_development_cors_allows_any_localhost_port(monkeypatch: pytest.MonkeyPatch, origin: str) -> None:
    monkeypatch.setattr(settings, "environment", "development")
    monkeypatch.setattr(settings, "cors_allow_tunnels", False)
    client = TestClient(create_app())

    response = _preflight(client, origin)

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_development_cors_allows_tunnels_only_when_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "environment", "development")
    monkeypatch.setattr(settings, "cors_allow_tunnels", True)
    client = TestClient(create_app())

    assert _preflight(client, "https://my-tunnel.ngrok-free.app").status_code == 200
    assert _preflight(client, "https://abcd.lhr.life").status_code == 200
    assert _preflight(client, "https://x.onrender.com").status_code == 400
    assert _preflight(client, "https://my-tunnel.ngrok-free.app.attacker.com").status_code == 400


def test_production_ignores_tunnel_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "cors_allow_tunnels", True)
    monkeypatch.setattr(settings, "cors_origins_str", "https://trusted-frontend.onrender.com")
    client = TestClient(create_app())

    assert _preflight(client, "https://my-tunnel.ngrok-free.app").status_code == 400
    assert _preflight(client, "http://localhost:5173").status_code == 400
    assert client.get("/health", headers={"Host": "my-tunnel.ngrok-free.app"}).status_code == 400


def test_trusted_hosts_exclude_tunnel_and_render_wildcards_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "environment", "development")
    monkeypatch.setattr(settings, "cors_allow_tunnels", False)
    monkeypatch.delenv("RENDER_EXTERNAL_HOSTNAME", raising=False)
    client = TestClient(create_app())

    assert client.get("/health", headers={"Host": "localhost"}).status_code == 200
    assert client.get("/health", headers={"Host": "attacker.onrender.com"}).status_code == 400
    assert client.get("/health", headers={"Host": "attacker.ngrok-free.app"}).status_code == 400


def test_trusted_hosts_accept_tunnels_when_opted_in_and_own_render_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "environment", "development")
    monkeypatch.setattr(settings, "cors_allow_tunnels", True)
    monkeypatch.setenv("RENDER_EXTERNAL_HOSTNAME", "my-api.onrender.com")
    client = TestClient(create_app())

    assert client.get("/health", headers={"Host": "my-tunnel.ngrok-free.app"}).status_code == 200
    assert client.get("/health", headers={"Host": "my-api.onrender.com"}).status_code == 200
    assert client.get("/health", headers={"Host": "attacker.onrender.com"}).status_code == 400
