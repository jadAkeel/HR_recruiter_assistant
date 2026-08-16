from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db_session
from app.models.candidate import Candidate
from app.models.user import User
from app.services.auth import decode_token, get_user_by_id
from sqlalchemy import select, true

from app.models.job import Job

security = HTTPBearer(auto_error=False)


def owned_resource_clause(model, user_id: str):
    """
    Keeps ownership-aware queries API-compatible while sharing all resources.
    """
    return true()


# Extract authenticated user from the Bearer token
async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    session: AsyncSession = Depends(get_db_session),
) -> User:
    """
    Authenticates the request and returns the current user.
    """
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    payload = decode_token(credentials.credentials)
    user_id = payload.get("sub") if payload else None
    if payload is None or payload.get("type") != "access" or not isinstance(user_id, str):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user = await get_user_by_id(session, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return user


def require_role(role: str):
    """
    Builds a dependency that requires one exact user role.
    """
    async def _check(user: User = Depends(get_current_user)) -> User:
        """
        Checks the current user role for a FastAPI dependency.
        """
        if user.role != role:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
        return user
    return _check


def require_any_role(*roles: str):
    """
    Builds a dependency that accepts any listed user role.
    """
    allowed = {role.lower() for role in roles}

    async def _check(user: User = Depends(get_current_user)) -> User:
        """
        Checks the current user role for a FastAPI dependency.
        """
        if user.role.lower() not in allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
        return user

    return _check


async def ensure_candidate_access(session: AsyncSession, user: User, candidate_id: str) -> None:
    """Checks that an authenticated user requested an existing candidate."""

    result = await session.execute(
        select(Candidate.id).where(Candidate.id == candidate_id)
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate not found")


async def ensure_job_access(session: AsyncSession, user: User, job_id: str) -> Job:
    """Loads an existing job for an authenticated user."""
    result = await session.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job
