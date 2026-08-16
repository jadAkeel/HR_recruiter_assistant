import uuid
import asyncio

from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from app.core.config import settings
from app.core.db import SessionLocal, init_db
from app.main import create_app
from app.models.user import User


async def _clear_users() -> None:
    """
    Removes user rows so owner bootstrap behavior can be tested deterministically.
    """
    await init_db()
    async with SessionLocal() as session:
        await session.execute(delete(User))
        await session.commit()


def test_register_and_login() -> None:
    """
    Checks that register and login.
    """
    app = create_app()
    with TestClient(app) as client:
        email = f"test-{uuid.uuid4().hex[:8]}@example.com"

        register_payload = {
            "email": email,
            "password": "strongpass123",
            "full_name": "Test User",
        }
        resp = client.post("/api/v1/auth/register", json=register_payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["email"] == email
        assert data["role"] == "owner"
        assert "id" in data

        login_payload = {"email": email, "password": "strongpass123"}
        resp = client.post("/api/v1/auth/login", json=login_payload)
        assert resp.status_code == 200
        tokens = resp.json()
        assert "access_token" in tokens
        assert "refresh_token" in tokens

        resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {tokens['access_token']}"})
        assert resp.status_code == 200
        me = resp.json()
        assert me["email"] == email


def test_first_registered_user_can_be_owner_when_enabled(monkeypatch) -> None:
    """
    Checks that deployments can bootstrap an owner from the first registered account.
    """
    asyncio.run(_clear_users())
    monkeypatch.setattr(settings, "first_user_owner_enabled", True)

    app = create_app()
    with TestClient(app) as client:
        email = f"owner-{uuid.uuid4().hex[:8]}@example.com"
        resp = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": "strongpass123", "full_name": "Owner User"},
        )

        assert resp.status_code == 201
        assert resp.json()["role"] == "owner"


def test_initial_owner_can_be_created_from_settings(monkeypatch) -> None:
    """
    Checks that a configured owner account is available after app startup.
    """
    asyncio.run(_clear_users())
    email = f"initial-owner-{uuid.uuid4().hex[:8]}@example.com"
    password = "strongpass123"
    monkeypatch.setattr(settings, "initial_owner_email", email)
    monkeypatch.setattr(settings, "initial_owner_password", password)
    monkeypatch.setattr(settings, "initial_owner_full_name", "Initial Owner")

    app = create_app()
    with TestClient(app) as client:
        login_resp = client.post("/api/v1/auth/login", json={"email": email, "password": password})
        assert login_resp.status_code == 200

        token = login_resp.json()["access_token"]
        me_resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me_resp.status_code == 200
        assert me_resp.json()["role"] == "owner"


def test_legacy_non_owner_authenticates_as_owner_and_is_persisted() -> None:
    """
    Checks that authentication normalizes a legacy role before granting access.
    """
    app = create_app()
    with TestClient(app) as client:
        email = f"candidate-{uuid.uuid4().hex[:8]}@example.com"
        register_payload = {
            "email": email,
            "password": "strongpass123",
            "full_name": "Candidate User",
        }
        assert client.post("/api/v1/auth/register", json=register_payload).status_code == 201

        async def _demote_legacy_user() -> None:
            """
            Simulates an environment where the owner-role migration has not run yet.
            """
            await init_db()
            async with SessionLocal() as session:
                user_res = await session.execute(select(User).where(User.email == email))
                legacy_user = user_res.scalar_one()
                legacy_user.role = "candidate"
                await session.commit()

        asyncio.run(_demote_legacy_user())

        login_resp = client.post("/api/v1/auth/login", json={"email": email, "password": "strongpass123"})
        assert login_resp.status_code == 200
        token = login_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        me_resp = client.get("/api/v1/auth/me", headers=headers)
        assert me_resp.status_code == 200
        assert me_resp.json()["role"] == "owner"

        first_bootstrap = client.post("/api/v1/auth/bootstrap-admin", headers=headers)
        second_bootstrap = client.post("/api/v1/auth/bootstrap-admin", headers=headers)
        assert first_bootstrap.status_code == 200
        assert second_bootstrap.status_code == 200
        assert second_bootstrap.json()["role"] == "owner"

        role_update = client.patch(
            f"/api/v1/auth/users/{me_resp.json()['id']}/role",
            json={"role": "candidate"},
            headers=headers,
        )
        assert role_update.status_code == 200
        assert role_update.json()["role"] == "owner"

    async def _load_role() -> str:
        async with SessionLocal() as session:
            user = (await session.execute(select(User).where(User.email == email))).scalar_one()
            return user.role

    assert asyncio.run(_load_role()) == "owner"
