from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db_session
from app.core.config import settings
from app.models.candidate import Candidate
from app.models.user import User
from app.services.auth import decode_token, get_user_by_id
from sqlalchemy import or_, select

from app.models.job import Job

security = HTTPBearer(auto_error=False)
STAFF_ROLES = {"owner", "admin", "recruiter"}


def owned_resource_clause(model, user_id: str):
    """
    Scopes resources strictly to their owning user.

    Legacy rows without an owner must be backfilled explicitly. Treating NULL
    as shared would expose those rows to every account.
    """
    if settings.allow_unowned_resources:
        return or_(model.created_by_user_id == user_id, model.created_by_user_id.is_(None))
    return model.created_by_user_id == user_id


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
    """Allow staff users to access only their own candidates and candidates their own CV row."""

    role = user.role.lower()
    if role in STAFF_ROLES:
        result = await session.execute(
            select(Candidate.id).where(
                Candidate.id == candidate_id,
                owned_resource_clause(Candidate, user.id),
            )
        )
        if result.scalar_one_or_none() is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate not found")
        return
    if role != "candidate":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    result = await session.execute(
        select(Candidate.id).where(
            Candidate.id == candidate_id,
            or_(Candidate.created_by_user_id == user.id, Candidate.email == user.email),
        )
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")


async def ensure_job_access(session: AsyncSession, user: User, job_id: str) -> Job:
    """Loads a job owned by the current staff user."""
    if user.role.lower() not in STAFF_ROLES:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    result = await session.execute(
        select(Job).where(Job.id == job_id, owned_resource_clause(Job, user.id))
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job
