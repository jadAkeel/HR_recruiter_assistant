"""Normalize all user roles to owner.

Revision ID: 0007_all_users_owner
Revises: 0006_cv_binary_storage
Create Date: 2026-08-16
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0007_all_users_owner"
down_revision = "0006_cv_binary_storage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("UPDATE users SET role = 'owner' WHERE role <> 'owner'"))


def downgrade() -> None:
    # Previous roles cannot be reconstructed safely.
    pass
