"""Add per-user ownership for recruiter data.

Revision ID: 0005_user_data_isolation
Revises: 0004_audit_evidence_versioning
Create Date: 2026-07-25
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0005_user_data_isolation"
down_revision = "0004_audit_evidence_versioning"
branch_labels = None
depends_on = None


def _has_column(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    if not sa.inspect(bind).has_table(table_name):
        return False
    return column_name in {column["name"] for column in sa.inspect(bind).get_columns(table_name)}


def _add_owned_column(table_name: str) -> None:
    if not _has_column(table_name, "created_by_user_id"):
        op.add_column(
            table_name,
            sa.Column("created_by_user_id", sa.String(length=36), nullable=True),
        )
    bind = op.get_bind()
    index_name = f"ix_{table_name}_created_by_user_id"
    indexes = {index["name"] for index in sa.inspect(bind).get_indexes(table_name)}
    if index_name not in indexes:
        op.create_index(index_name, table_name, ["created_by_user_id"])


def upgrade() -> None:
    _add_owned_column("jobs")
    _add_owned_column("candidates")
    _add_owned_column("skill_feedback")

    # Existing rows belong to the first staff account so migration does not expose
    # legacy data to every newly registered staff user.
    bind = op.get_bind()
    owner_id = bind.execute(
        sa.text(
            "SELECT id FROM users "
            "WHERE lower(role) IN ('owner', 'admin', 'recruiter') "
            "ORDER BY id LIMIT 1"
        )
    ).scalar_one_or_none()
    if owner_id is not None:
        for table_name in ("jobs", "candidates", "skill_feedback"):
            bind.execute(
                sa.text(
                    f"UPDATE {table_name} SET created_by_user_id = :owner_id "
                    "WHERE created_by_user_id IS NULL"
                ),
                {"owner_id": owner_id},
            )


def downgrade() -> None:
    for table_name in ("skill_feedback", "candidates", "jobs"):
        bind = op.get_bind()
        index_name = f"ix_{table_name}_created_by_user_id"
        indexes = {index["name"] for index in sa.inspect(bind).get_indexes(table_name)}
        if index_name in indexes:
            op.drop_index(index_name, table_name=table_name)
        if _has_column(table_name, "created_by_user_id"):
            op.drop_column(table_name, "created_by_user_id")
