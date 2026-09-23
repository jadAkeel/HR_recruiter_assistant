from __future__ import annotations

from sqlalchemy import JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.config import settings
from app.core.db import engine
from app.models.base import Base

try:
    from pgvector.sqlalchemy import Vector
except ImportError:  # pragma: no cover
    Vector = None

_USE_VECTOR = engine.dialect.name == "postgresql" and Vector is not None


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # Stable identity for seeded documents ("skill:nlp", "policy:hiring-best-practices").
    # User-ingested documents leave it NULL.
    doc_key: Mapped[str | None] = mapped_column(String(128), nullable=True, unique=True, index=True)
    title: Mapped[str] = mapped_column(String(255))
    content: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(50), index=True)
    tags: Mapped[list[str]] = mapped_column(JSON)
    indicators: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    not_sufficient: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    embedding_multilingual: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)

    if _USE_VECTOR:
        embedding_vector: Mapped[list[float] | None] = mapped_column(Vector(settings.embedding_dimension), nullable=True)
    else:
        # Only populated on PostgreSQL; SQLite ranks the JSON embedding in numpy.
        embedding_vector: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
