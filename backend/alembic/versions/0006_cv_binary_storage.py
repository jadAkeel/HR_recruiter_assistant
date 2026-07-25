"""Store uploaded CV files in the database.

Revision ID: 0006_cv_binary_storage
Revises: 0005_user_data_isolation
Create Date: 2026-07-25
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0006_cv_binary_storage"
down_revision = "0005_user_data_isolation"
branch_labels = None
depends_on = None


def _has_column(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    if not sa.inspect(bind).has_table(table_name):
        return False
    return column_name in {column["name"] for column in sa.inspect(bind).get_columns(table_name)}


def upgrade() -> None:
    if not _has_column("candidates", "cv_file_name"):
        op.add_column("candidates", sa.Column("cv_file_name", sa.String(length=255), nullable=True))
    if not _has_column("candidates", "cv_content"):
        op.add_column("candidates", sa.Column("cv_content", sa.LargeBinary(), nullable=True))


def downgrade() -> None:
    if _has_column("candidates", "cv_content"):
        op.drop_column("candidates", "cv_content")
    if _has_column("candidates", "cv_file_name"):
        op.drop_column("candidates", "cv_file_name")
