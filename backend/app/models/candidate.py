from __future__ import annotations

from sqlalchemy import JSON, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Candidate(Base):
    __tablename__ = "candidates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_by_user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=True, index=True
    )
    full_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    email: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    phone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    skills: Mapped[list[str]] = mapped_column(JSON)
    skills_detailed: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    experience: Mapped[list[str]] = mapped_column(JSON)
    experience_entries: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    education: Mapped[list[str]] = mapped_column(JSON)
    education_entries: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    projects: Mapped[list[str]] = mapped_column(JSON)
    negative_skills: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    learning_skills: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    uncatalogued_skills: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    total_years_experience: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_text: Mapped[str] = mapped_column(Text)
