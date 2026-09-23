"""Add keyed, hash-tracked skill knowledge columns for RAG enrichment.

Revision ID: 0008_rag_skill_knowledge
Revises: 0007_all_users_owner
Create Date: 2026-09-23
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from app.core.config import settings

try:
    from pgvector.sqlalchemy import Vector
except ImportError:  # pragma: no cover
    Vector = None

revision = "0008_rag_skill_knowledge"
down_revision = "0007_all_users_owner"
branch_labels = None
depends_on = None

_COLUMNS = ("doc_key", "indicators", "not_sufficient", "content_hash", "embedding_model", "embedding_vector")


def _has_column(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    if not sa.inspect(bind).has_table(table_name):
        return False
    columns = {column["name"] for column in sa.inspect(bind).get_columns(table_name)}
    return column_name in columns


def _add_column_if_missing(table_name: str, column: sa.Column) -> None:
    if not _has_column(table_name, column.name):
        op.add_column(table_name, column)


def _vector_column_type() -> sa.types.TypeEngine:
    if op.get_bind().dialect.name == "postgresql" and Vector is not None:
        return Vector(settings.embedding_dimension)
    return sa.JSON()


def upgrade() -> None:
    table = "knowledge_documents"
    _add_column_if_missing(table, sa.Column("doc_key", sa.String(length=128), nullable=True))
    _add_column_if_missing(table, sa.Column("indicators", sa.JSON(), nullable=True))
    _add_column_if_missing(table, sa.Column("not_sufficient", sa.JSON(), nullable=True))
    _add_column_if_missing(table, sa.Column("content_hash", sa.String(length=64), nullable=True))
    _add_column_if_missing(table, sa.Column("embedding_model", sa.String(length=255), nullable=True))
    _add_column_if_missing(table, sa.Column("embedding_vector", _vector_column_type(), nullable=True))
    indexes = {index["name"] for index in sa.inspect(op.get_bind()).get_indexes(table)}
    if "ix_knowledge_documents_doc_key" not in indexes:
        op.create_index("ix_knowledge_documents_doc_key", table, ["doc_key"], unique=True)


def downgrade() -> None:
    table = "knowledge_documents"
    indexes = {index["name"] for index in sa.inspect(op.get_bind()).get_indexes(table)}
    if "ix_knowledge_documents_doc_key" in indexes:
        op.drop_index("ix_knowledge_documents_doc_key", table_name=table)
    for column in _COLUMNS:
        if _has_column(table, column):
            op.drop_column(table, column)
