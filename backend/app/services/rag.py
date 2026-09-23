from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import db
from app.core.config import settings
from app.models.knowledge import KnowledgeDocument
from app.schemas.rag import DocumentItem, IngestResponse, QueryResponse
from app.services.embedding import (
    detect_language,
    embedding_model_name_for_provider,
    get_multilingual_embedding_service,
    get_standard_embedding_service,
)
from app.services.skill_catalog import _resolve_data_path, build_skill_pattern, normalize_skill_name

logger = logging.getLogger(__name__)

SKILL_DOC_PREFIX = "skill:"

INITIAL_KNOWLEDGE: list[dict[str, Any]] = [
    {
        "doc_key": "skill:python",
        "title": "Python Skill Definition",
        "content": "Python is a high-level, interpreted programming language known for readability and versatility. Used in web development, data science, machine learning, and automation. Key concepts: dynamic typing, garbage collection, list comprehensions, generators, decorators, context managers.",
        "category": "skill",
        "tags": ["python", "programming", "backend"],
    },
    {
        "doc_key": "skill:nlp",
        "title": "NLP Skill Definition",
        "content": "Natural Language Processing involves enabling computers to understand, interpret, and generate human language. Key techniques: tokenization, stemming, named entity recognition, sentiment analysis, word embeddings, transformer architectures (BERT, GPT), sequence labeling, text classification.",
        "category": "skill",
        "tags": ["nlp", "machine learning", "ai"],
    },
    {
        "doc_key": "policy:seniority-evaluation-guidelines",
        "title": "Seniority Evaluation Guidelines",
        "content": "Junior: 0-2 years experience, works under guidance, foundational knowledge. Mid-level: 2-5 years, independent contributor, solid problem-solving. Senior: 5+ years, technical leadership, architectural decisions, mentoring.",
        "category": "policy",
        "tags": ["seniority", "evaluation", "hiring"],
    },
    {
        "doc_key": "policy:hiring-best-practices",
        "title": "Hiring Best Practices",
        "content": "Focus on both technical skills and cultural fit. Evaluate problem-solving approach over memorized answers. Consider diversity and inclusion. Use structured interviews with consistent scoring. Provide constructive feedback to all candidates.",
        "category": "policy",
        "tags": ["hiring", "best-practices", "recruitment"],
    },
    {
        "doc_key": "skill:docker",
        "title": "Docker Skill Definition",
        "content": "Docker is a containerization platform that enables packaging applications with their dependencies into containers. Key concepts: images, containers, Dockerfile, docker-compose, volumes, networks, multi-stage builds, container orchestration.",
        "category": "skill",
        "tags": ["docker", "devops", "containers"],
    },
]


@dataclass(frozen=True)
class SkillKnowledge:
    """Knowledge-base entry used to judge whether a skill is present by meaning."""

    skill: str
    definition: str
    indicators: tuple[str, ...] = ()
    not_sufficient: tuple[str, ...] = ()
    embedding: list[float] | None = None
    embedding_model: str | None = None
    doc_key: str | None = None


@dataclass
class KnowledgeSyncReport:
    """Outcome of one idempotent knowledge-base sync."""

    inserted: int = 0
    updated: int = 0
    reembedded: int = 0
    unchanged: int = 0
    embedding_failed: int = 0
    total: int = 0
    embedding_model: str = ""
    elapsed_ms: float = 0.0
    inserted_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the report without the potentially long key list.
        """
        data = asdict(self)
        data.pop("inserted_keys", None)
        return data


def skill_doc_key(skill: str) -> str:
    """
    Returns the stable document key for a canonical skill definition.
    """
    return f"{SKILL_DOC_PREFIX}{normalize_skill_name(skill)}"


def content_hash(content: str) -> str:
    """
    Hashes the text that gets embedded, so unchanged documents are never re-embedded.
    """
    return hashlib.sha256(str(content or "").encode("utf-8")).hexdigest()


def load_skill_knowledge_file(path: str | None = None) -> list[dict[str, Any]]:
    """
    Loads and validates umbrella-skill entries from the skill knowledge JSON file.
    """
    data_path = _resolve_data_path(path or settings.skill_knowledge_path)
    if not data_path.exists():
        logger.warning("Skill knowledge file not found", extra={"path": str(data_path)})
        return []
    with data_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    raw_entries = data.get("skills", []) if isinstance(data, dict) else data
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_entries or []:
        if not isinstance(raw, dict):
            continue
        skill = normalize_skill_name(str(raw.get("skill", "")))
        definition = " ".join(str(raw.get("definition", "")).split())
        if not skill or not definition or skill in seen:
            continue
        seen.add(skill)
        entries.append({
            "skill": skill,
            "definition": definition,
            "indicators": _clean_terms(raw.get("indicators")),
            "not_sufficient": _clean_terms(raw.get("not_sufficient")),
        })
    return entries


def _clean_terms(values: Any) -> list[str]:
    """
    Lowercases, trims and de-duplicates a list of evidence terms.
    """
    terms: list[str] = []
    for value in values or []:
        term = " ".join(str(value or "").strip().lower().split())
        if term and term not in terms:
            terms.append(term)
    return terms


def build_seed_documents(path: str | None = None) -> list[dict[str, Any]]:
    """
    Merges INITIAL_KNOWLEDGE with the skill knowledge file into seedable documents.

    File entries win for skills defined in both places, but keep the built-in
    title so databases seeded by the old endpoint are adopted instead of duplicated.
    """
    documents: dict[str, dict[str, Any]] = {}
    for doc in INITIAL_KNOWLEDGE:
        documents[doc["doc_key"]] = {
            "doc_key": doc["doc_key"],
            "title": doc["title"],
            "content": doc["content"],
            "category": doc["category"],
            "tags": list(doc["tags"]),
            "indicators": None,
            "not_sufficient": None,
        }
    for entry in load_skill_knowledge_file(path):
        key = skill_doc_key(entry["skill"])
        builtin = documents.get(key)
        documents[key] = {
            "doc_key": key,
            "title": builtin["title"] if builtin else f"Skill Definition: {entry['skill']}",
            "content": entry["definition"],
            "category": "skill",
            "tags": [entry["skill"]],
            "indicators": entry["indicators"],
            "not_sufficient": entry["not_sufficient"],
        }
    return list(documents.values())


def _uses_pgvector() -> bool:
    """
    Checks whether knowledge embeddings live in a pgvector column on PostgreSQL.
    """
    return db.engine.dialect.name == "postgresql" and hasattr(KnowledgeDocument.embedding_vector, "cosine_distance")


def _store_embedding(document: KnowledgeDocument, embedding: list[float], model_name: str) -> None:
    """
    Stores a standard embedding and the metadata that says which content and model produced it.
    """
    document.embedding = embedding
    document.embedding_model = model_name
    document.content_hash = content_hash(document.content)
    if _uses_pgvector() and len(embedding) == settings.embedding_dimension:
        document.embedding_vector = embedding


def _should_embed_multilingual() -> bool:
    """
    Multilingual vectors load a second model, so only build them when requested.
    """
    return settings.use_multilingual_embedding or settings.rag_seed_multilingual


async def _embed_documents(documents: list[KnowledgeDocument], model_name: str) -> int:
    """
    Embeds documents in one batch per model; returns how many failed.
    """
    if not documents:
        return 0
    failed = 0
    try:
        vectors = await get_standard_embedding_service().embed([doc.content for doc in documents])
        for document, vector in zip(documents, vectors):
            _store_embedding(document, vector, model_name)
    except Exception:
        logger.warning("Knowledge embedding batch failed", extra={"count": len(documents)}, exc_info=True)
        for document in documents:
            # Never keep a vector that belongs to older content.
            if document.content_hash != content_hash(document.content):
                document.embedding = None
                document.embedding_vector = None
                document.content_hash = None
        failed = len(documents)

    if _should_embed_multilingual():
        try:
            vectors = await get_multilingual_embedding_service().embed([doc.content for doc in documents])
            for document, vector in zip(documents, vectors):
                document.embedding_multilingual = vector
        except Exception:
            logger.warning("Multilingual knowledge embedding batch failed", extra={"count": len(documents)})
    return failed


async def sync_knowledge_base(
    session: AsyncSession,
    *,
    path: str | None = None,
    _retry: bool = True,
) -> KnowledgeSyncReport:
    """
    Idempotently upserts built-in knowledge and re-embeds only changed documents.

    A document is re-embedded when its content hash or the embedding model
    changed; metadata such as indicators is updated in place without embedding.
    """
    started = time.perf_counter()
    model_name = embedding_model_name_for_provider()
    report = KnowledgeSyncReport(embedding_model=model_name)
    seed_docs = build_seed_documents(path)
    report.total = len(seed_docs)
    keys = [doc["doc_key"] for doc in seed_docs]
    titles = [doc["title"] for doc in seed_docs]

    result = await session.execute(
        select(KnowledgeDocument).where(
            or_(
                KnowledgeDocument.doc_key.in_(keys),
                KnowledgeDocument.doc_key.is_(None) & KnowledgeDocument.title.in_(titles),
            )
        )
    )
    rows = result.scalars().all()
    by_key = {row.doc_key: row for row in rows if row.doc_key}
    legacy_by_title = {row.title: row for row in rows if not row.doc_key}

    to_embed: list[KnowledgeDocument] = []
    for doc in seed_docs:
        row = by_key.get(doc["doc_key"]) or legacy_by_title.pop(doc["title"], None)
        if row is None:
            row = KnowledgeDocument(id=str(uuid.uuid4()), **doc)
            session.add(row)
            to_embed.append(row)
            report.inserted += 1
            report.inserted_keys.append(doc["doc_key"])
            continue

        metadata_changed = False
        for attr in ("doc_key", "title", "category", "tags", "indicators", "not_sufficient"):
            if getattr(row, attr) != doc[attr]:
                setattr(row, attr, doc[attr])
                metadata_changed = True
        if row.content != doc["content"]:
            row.content = doc["content"]
            metadata_changed = True

        needs_embedding = (
            row.embedding is None
            or row.content_hash != content_hash(row.content)
            or row.embedding_model != model_name
            or (_should_embed_multilingual() and row.embedding_multilingual is None)
        )
        if needs_embedding:
            to_embed.append(row)
            report.reembedded += 1
        elif _uses_pgvector() and row.embedding_vector is None and len(row.embedding or []) == settings.embedding_dimension:
            row.embedding_vector = row.embedding
            metadata_changed = True
        if metadata_changed or needs_embedding:
            report.updated += 1
        else:
            report.unchanged += 1

    report.embedding_failed = await _embed_documents(to_embed, model_name)
    try:
        await session.commit()
    except IntegrityError:
        # Another process (API vs. worker) inserted the same doc_key first.
        await session.rollback()
        if not _retry:
            raise
        logger.info("Knowledge base sync raced with another process; retrying once")
        return await sync_knowledge_base(session, path=path, _retry=False)
    report.elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    return report


async def seed_knowledge_base_on_startup() -> None:
    """
    Background startup task: syncs the knowledge base and logs what changed.
    """
    from app.core.db import SessionLocal

    try:
        async with SessionLocal() as session:
            report = await sync_knowledge_base(session)
        logger.info(
            "RAG knowledge base synced: %d documents (%d inserted, %d re-embedded, %d unchanged, "
            "%d embedding failures) with %s in %.0f ms",
            report.total,
            report.inserted,
            report.reembedded,
            report.unchanged,
            report.embedding_failed,
            report.embedding_model,
            report.elapsed_ms,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("RAG knowledge base sync failed; enrichment will use existing documents", exc_info=True)


async def ingest_knowledge_base(session: AsyncSession) -> int:
    """
    Seeds the database with built-in knowledge documents; returns how many were inserted.
    """
    report = await sync_knowledge_base(session)
    return report.inserted


async def add_document(
    session: AsyncSession,
    title: str,
    content: str,
    category: str,
    tags: list[str],
    indicators: list[str] | None = None,
    not_sufficient: list[str] | None = None,
) -> IngestResponse:
    """
    Adds one knowledge document and its embedding to the database.
    """
    document = KnowledgeDocument(
        id=str(uuid.uuid4()),
        title=title,
        content=content,
        category=category,
        tags=tags,
        indicators=_clean_terms(indicators) or None,
        not_sufficient=_clean_terms(not_sufficient) or None,
    )
    await _embed_documents([document], embedding_model_name_for_provider())
    session.add(document)
    await session.commit()
    return IngestResponse(document_id=document.id, title=title, category=category)


async def query_knowledge(
    session: AsyncSession,
    query: str,
    category: str | None = None,
    top_k: int = 5,
) -> QueryResponse:
    """
    Ranks knowledge documents by embedding similarity to a query.

    PostgreSQL ranks inside the database with pgvector; SQLite (and the
    multilingual route, which has no vector column) scores in numpy.
    """
    query_language = detect_language(query)
    use_multilingual = query_language == "ar"

    try:
        embedder = get_multilingual_embedding_service() if use_multilingual else get_standard_embedding_service()
        query_emb = (await embedder.embed([query]))[0]
    except Exception:
        logger.warning("Knowledge query embedding failed")
        return QueryResponse(query=query, results=[], query_language=query_language)

    if not use_multilingual and _uses_pgvector() and len(query_emb) == settings.embedding_dimension:
        scored = await _query_knowledge_pgvector(session, query_emb, category, top_k)
    else:
        scored = await _query_knowledge_in_memory(session, query_emb, category, top_k, use_multilingual)

    results = [
        DocumentItem(
            document_id=doc.id,
            title=doc.title,
            content=doc.content,
            category=doc.category,
            score=round(score, 4),
        )
        for doc, score in scored
    ]
    return QueryResponse(query=query, results=results, query_language=query_language)


def pgvector_knowledge_statement(
    query_embedding: list[float],
    category: str | None,
    top_k: int,
    vector_column: Any | None = None,
):
    """
    Builds the pgvector nearest-neighbour query (the same cosine_distance
    pattern VectorStore uses) so ranking and LIMIT happen inside PostgreSQL.
    """
    column = vector_column if vector_column is not None else KnowledgeDocument.embedding_vector
    distance = column.cosine_distance(query_embedding).label("distance")
    stmt = select(KnowledgeDocument, distance).where(column.is_not(None))
    if category:
        stmt = stmt.where(KnowledgeDocument.category == category)
    return stmt.order_by(distance).limit(top_k)


async def _query_knowledge_pgvector(
    session: AsyncSession,
    query_embedding: list[float],
    category: str | None,
    top_k: int,
) -> list[tuple[KnowledgeDocument, float]]:
    """
    Lets pgvector order documents by cosine distance and return only the top rows.
    """
    result = await session.execute(pgvector_knowledge_statement(query_embedding, category, top_k))
    return [(row[0], float(1.0 - row[1])) for row in result.all()]


async def _query_knowledge_in_memory(
    session: AsyncSession,
    query_embedding: list[float],
    category: str | None,
    top_k: int,
    use_multilingual: bool,
) -> list[tuple[KnowledgeDocument, float]]:
    """
    Scores knowledge documents in numpy for databases without pgvector.
    """
    stmt = select(KnowledgeDocument)
    if category:
        stmt = stmt.where(KnowledgeDocument.category == category)
    result = await session.execute(stmt)
    documents: list[KnowledgeDocument] = []
    vectors: list[list[float]] = []
    for doc in result.scalars().all():
        # Arabic queries rank against multilingual vectors, English against standard ones.
        emb = doc.embedding_multilingual if use_multilingual else doc.embedding
        if emb is None:
            emb = doc.embedding if use_multilingual else doc.embedding_multilingual
        if emb is None or len(emb) != len(query_embedding):
            continue
        documents.append(doc)
        vectors.append(emb)
    if not documents:
        return []

    matrix = np.asarray(vectors, dtype=np.float32)
    query = np.asarray(query_embedding, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query) or 1.0)
    norms[norms == 0] = 1.0
    scores = matrix @ query / norms
    order = np.argsort(-scores)[:top_k]
    return [(documents[i], float(scores[i])) for i in order]


def _to_skill_knowledge(skill: str, document: KnowledgeDocument) -> SkillKnowledge:
    """
    Converts a knowledge row into the immutable view used by the matcher.
    """
    return SkillKnowledge(
        skill=skill,
        definition=document.content,
        indicators=tuple(_clean_terms(document.indicators)),
        not_sufficient=tuple(_clean_terms(document.not_sufficient)),
        embedding=list(document.embedding) if document.embedding else None,
        embedding_model=document.embedding_model,
        doc_key=document.doc_key,
    )


async def get_skill_knowledge(
    skills: list[str],
    session: AsyncSession | None = None,
) -> dict[str, SkillKnowledge]:
    """
    Looks up knowledge for canonical skill names: keyed by doc_key first, then
    by exact tag or token-bounded title match for user-ingested documents.
    """
    if session is None:
        from app.core.db import SessionLocal

        async with SessionLocal() as managed_session:
            return await get_skill_knowledge(skills, managed_session)

    normalized: list[str] = []
    for skill in skills:
        name = normalize_skill_name(str(skill))
        if name and name not in normalized:
            normalized.append(name)
    if not normalized:
        return {}

    keyed = {skill_doc_key(skill): skill for skill in normalized}
    result = await session.execute(select(KnowledgeDocument).where(KnowledgeDocument.doc_key.in_(list(keyed))))
    knowledge = {
        keyed[document.doc_key]: _to_skill_knowledge(keyed[document.doc_key], document)
        for document in result.scalars().all()
    }

    unresolved = [skill for skill in normalized if skill not in knowledge]
    if unresolved:
        result = await session.execute(
            select(KnowledgeDocument).where(
                KnowledgeDocument.category == "skill",
                KnowledgeDocument.doc_key.is_(None),
            )
        )
        documents = result.scalars().all()
        for skill in unresolved:
            best: tuple[KnowledgeDocument, int] | None = None
            for document in documents:
                tags = {normalize_skill_name(str(tag)) for tag in document.tags or []}
                if skill in tags:
                    score = 2
                elif build_skill_pattern(skill).search((document.title or "").lower()):
                    score = 1
                else:
                    continue
                if best is None or score > best[1]:
                    best = (document, score)
            if best:
                knowledge[skill] = _to_skill_knowledge(skill, best[0])
    return knowledge


async def get_skill_definitions(
    skills: list[str],
    session: AsyncSession | None = None,
) -> dict[str, str]:
    """
    Returns knowledge-base definitions for requested skills.
    """
    knowledge = await get_skill_knowledge(skills, session=session)
    return {skill: entry.definition for skill, entry in knowledge.items()}
