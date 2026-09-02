import asyncio
import getpass
import os
from backend.app.core.db import get_db_session
from backend.app.models.user import User
from backend.app.services.auth import hash_password
from sqlalchemy import select

async def fix():
    """
    Updates the local test user password helper.
    """
    email = os.getenv("RESET_USER_EMAIL") or input("User email: ").strip()
    password = os.getenv("RESET_USER_PASSWORD") or getpass.getpass("New password: ")
    if not email or not password:
        raise ValueError("Email and password are required")

    async for session in get_db_session():
        result = await session.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        if user is None:
            raise ValueError(f"No user found for {email}")
        user.password_hash = hash_password(password)
        await session.commit()
        print(f"Password updated for {email}")
        break

asyncio.run(fix())
