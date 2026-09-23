import json
import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy import delete, func, or_, select
from sqlalchemy.dialects import postgresql

from app.core.config import settings
from app.core.db import SessionLocal, init_db
from app.models.candidate import Candidate
from app.models.job import Job
from app.models.knowledge import KnowledgeDocument
from app.services import rag
from app.services.hybrid_matcher import HybridMatchingEngine, SkillMatchResult
from app.services.rag import (
    SkillKnowledge,
    content_hash,
    get_skill_definitions,
    get_skill_knowledge,
    load_skill_knowledge_file,
    pgvector_knowledge_statement,
    skill_doc_key,
    sync_knowledge_base,
)
from app.services.skill_catalog import build_term_pattern, normalize_skill_name

BACKEND_DIR = Path(__file__).resolve().parents[1]
VAGUE_TERMS = {"data", "analysis", "model", "models", "ai", "software", "development", "experience", "tools", "system"}

NLP = SkillKnowledge(
    skill="nlp",
    definition="Natural language processing builds systems that analyze and generate human language text.",
    indicators=("bert", "tokenization", "named entity recognition", "spacy", "text classification", "sentiment analysis"),
    not_sufficient=("payment tokenization", "customer sentiment analysis"),
)
DEVOPS = SkillKnowledge(
    skill="devops",
    definition="DevOps automates building, testing, releasing and operating software.",
    indicators=("jenkins", "terraform", "ansible", "ci/cd", "vector database", "faiss"),
    not_sufficient=("git",),
)


def _candidate(text: str, **overrides) -> Candidate:
    fields = {
        "id": str(uuid.uuid4()),
        "full_name": "RAG Candidate",
        "email": f"{uuid.uuid4().hex[:8]}@example.com",
        "phone": None,
        "skills": [],
        "experience": [text],
        "education": [],
        "projects": [],
        "raw_text": text,
    }
    fields.update(overrides)
    return Candidate(**fields)


def _engine(*entries: SkillKnowledge) -> HybridMatchingEngine:
    engine = HybridMatchingEngine()
    engine._rag_knowledge_cache.update({entry.skill: entry for entry in entries})
    return engine


async def _enrich(engine: HybridMatchingEngine, skill: str, candidate: Candidate) -> SkillMatchResult:
    result = SkillMatchResult(missing_required=[skill])
    await engine._apply_rag_enrichment(result, candidate, session=None)
    return result


def _write_knowledge(path: Path, entries: list[dict]) -> str:
    path.write_text(json.dumps({"version": 1, "skills": entries}), encoding="utf-8")
    return str(path)


# --- knowledge base content -------------------------------------------------

def test_skill_knowledge_file_is_complete_and_specific() -> None:
    entries = load_skill_knowledge_file()
    assert len(entries) >= 60
    for entry in entries:
        assert entry["skill"] == normalize_skill_name(entry["skill"])
        assert 20 <= len(entry["definition"]) <= 600, entry["skill"]
        assert len(entry["indicators"]) >= 3, entry["skill"]
        assert entry["not_sufficient"], entry["skill"]
        assert not VAGUE_TERMS & set(entry["indicators"]), entry["skill"]
        assert not set(entry["indicators"]) & set(entry["not_sufficient"]), entry["skill"]
        assert entry["skill"] not in entry["indicators"], entry["skill"]
        # Alternate spellings ("express.js"/"expressjs") may share a canonical
        # key; the matcher counts them as one indicator.
        assert len(entry["indicators"]) == len(set(entry["indicators"])), entry["skill"]
        for term in entry["indicators"] + entry["not_sufficient"]:
            build_term_pattern(term)


def test_term_pattern_handles_plurals_versions_compounds_and_ambiguous_aliases() -> None:
    assert build_term_pattern("helm chart").search("wrote helm charts for 30 services")
    assert build_term_pattern("yolo").search("trained yolov5 detectors")
    assert build_term_pattern("bert").search("bert-based classifiers")
    assert not build_term_pattern("bert").search("roberta encoders")
    assert not build_term_pattern("docker").search("docker-compose files")
    assert not build_term_pattern("node.js").search("each node in the graph")
    assert not build_term_pattern("express.js").search("express my ideas clearly")
    assert build_term_pattern("scikit-learn").search("models in sklearn")


# --- idempotent seeding -----------------------------------------------------

@pytest.mark.asyncio
async def test_sync_is_idempotent_and_only_reembeds_changed_content(tmp_path, monkeypatch) -> None:
    await init_db()
    suffix = uuid.uuid4().hex[:8]
    alpha, beta = f"alpha skill {suffix}", f"beta skill {suffix}"
    entries = [
        {"skill": alpha, "definition": "Alpha skill builds alpha pipelines for testing.", "indicators": ["alpha one", "alpha two"], "not_sufficient": ["alpha lookalike"]},
        {"skill": beta, "definition": "Beta skill runs beta services in production.", "indicators": ["beta one", "beta two"], "not_sufficient": ["beta lookalike"]},
    ]
    path = _write_knowledge(tmp_path / "knowledge.json", entries)
    keys = [skill_doc_key(alpha), skill_doc_key(beta)]

    async def rows() -> dict[str, KnowledgeDocument]:
        async with SessionLocal() as session:
            result = await session.execute(select(KnowledgeDocument).where(KnowledgeDocument.doc_key.in_(keys)))
            return {row.doc_key: row for row in result.scalars().all()}

    async with SessionLocal() as session:
        first = await sync_knowledge_base(session, path=path)
    assert set(keys) <= set(first.inserted_keys)
    seeded = await rows()
    assert len(seeded) == 2
    assert all(row.embedding and row.content_hash == content_hash(row.content) for row in seeded.values())
    assert all(row.embedding_model == "hash" for row in seeded.values())

    async with SessionLocal() as session:
        second = await sync_knowledge_base(session, path=path)
    assert second.inserted == 0
    assert second.reembedded == 0
    assert second.unchanged == second.total
    async with SessionLocal() as session:
        count = await session.scalar(select(func.count()).select_from(KnowledgeDocument).where(KnowledgeDocument.doc_key.in_(keys)))
    assert count == 2

    # Changed definition -> only that document is re-embedded.
    entries[0]["definition"] = "Alpha skill now builds streaming alpha pipelines."
    # Changed indicators only -> metadata update without a new embedding.
    entries[1]["indicators"] = ["beta one", "beta three"]
    _write_knowledge(tmp_path / "knowledge.json", entries)
    async with SessionLocal() as session:
        third = await sync_knowledge_base(session, path=path)
    after = await rows()
    assert third.inserted == 0
    assert third.reembedded == 1
    assert after[keys[0]].content_hash == content_hash(entries[0]["definition"])
    assert after[keys[0]].embedding != seeded[keys[0]].embedding
    assert after[keys[1]].embedding == seeded[keys[1]].embedding
    assert after[keys[1]].indicators == ["beta one", "beta three"]

    # A different embedding model re-embeds every seeded document.
    monkeypatch.setattr(rag, "embedding_model_name_for_provider", lambda *args, **kwargs: "other-model")
    async with SessionLocal() as session:
        fourth = await sync_knowledge_base(session, path=path)
    assert fourth.reembedded == fourth.total
    assert all(row.embedding_model == "other-model" for row in (await rows()).values())


@pytest.mark.asyncio
async def test_sync_adopts_legacy_seeded_rows_instead_of_duplicating() -> None:
    await init_db()
    async with SessionLocal() as session:
        await session.execute(delete(KnowledgeDocument).where(
            or_(KnowledgeDocument.doc_key == "skill:nlp", KnowledgeDocument.title == "NLP Skill Definition")
        ))
        # Row created by the old POST /rag/seed: no doc_key, no hash.
        session.add(KnowledgeDocument(
            id=str(uuid.uuid4()), title="NLP Skill Definition", content="Old NLP text.",
            category="skill", tags=["nlp"], embedding=None,
        ))
        await session.commit()
        await sync_knowledge_base(session)
        await sync_knowledge_base(session)
        result = await session.execute(select(KnowledgeDocument).where(
            or_(KnowledgeDocument.doc_key == "skill:nlp", KnowledgeDocument.title == "NLP Skill Definition")
        ))
        documents = result.scalars().all()
    assert len(documents) == 1
    assert documents[0].doc_key == "skill:nlp"
    assert "bert" in documents[0].indicators


@pytest.mark.asyncio
async def test_seed_endpoint_function_and_startup_task_share_sync(monkeypatch) -> None:
    from app.main import _start_knowledge_sync

    monkeypatch.setattr(settings, "rag_auto_seed", False)
    assert _start_knowledge_sync() is None

    await init_db()
    await rag.seed_knowledge_base_on_startup()
    async with SessionLocal() as session:
        knowledge = await get_skill_knowledge(["NLP", "Machine Learning", "not-a-seeded-skill"], session=session)
        assert await rag.ingest_knowledge_base(session) == 0
    assert set(knowledge) == {"nlp", "machine learning"}
    assert "tokenization" in knowledge["nlp"].indicators
    assert knowledge["nlp"].doc_key == "skill:nlp"


@pytest.mark.asyncio
async def test_skill_definition_fallback_uses_tags_and_token_bounded_titles() -> None:
    await init_db()
    async with SessionLocal() as session:
        session.add(KnowledgeDocument(
            id=str(uuid.uuid4()), title="Zig Toolchain Notes", content="Zig builds.", category="skill", tags=["zig"],
        ))
        await session.commit()
        definitions = await get_skill_definitions(["zig", "r"], session=session)
    assert definitions["zig"] == "Zig builds."
    # "r" must not substring-match titles such as "Docker Skill Definition".
    assert "r" not in definitions


def test_pgvector_statement_orders_by_cosine_distance_in_sql() -> None:
    from pgvector.sqlalchemy import Vector

    vector_column = sa.Table("kd_probe", sa.MetaData(), sa.Column("embedding_vector", Vector(3))).c.embedding_vector
    stmt = pgvector_knowledge_statement([0.1, 0.2, 0.3], "skill", 5, vector_column=vector_column)
    sql = str(stmt.compile(dialect=postgresql.dialect()))
    assert "<=>" in sql
    assert "ORDER BY distance" in sql
    assert "LIMIT" in sql


# --- evidence logic -----------------------------------------------------------

@pytest.mark.asyncio
async def test_two_distinct_indicators_grant_capped_partial_credit(monkeypatch) -> None:
    monkeypatch.setattr(settings, "rag_match_confidence", 0.95)
    candidate = _candidate("Fine-tuned BERT models and wrote a custom tokenization pipeline for support tickets.")
    result = await _enrich(_engine(NLP), "nlp", candidate)

    assert result.rag_enriched_skills == ["nlp"]
    assert result.matched_required[0].match_type == "rag"
    assert result.matched_required[0].confidence == pytest.approx(0.60)
    evidence = result.rag_evidence["nlp"]
    assert set(evidence["indicators"]) == {"bert", "tokenization"}
    assert "BERT" in evidence["best_chunk"]
    assert evidence["similarity"] is None  # hash provider: no semantic evidence
    assert "distinct indicators" in evidence["rule"]


@pytest.mark.asyncio
async def test_single_indicator_without_semantic_evidence_is_refused() -> None:
    result = await _enrich(_engine(NLP), "nlp", _candidate("Built a text classification service for emails."))
    assert result.rag_enriched_skills == []
    assert result.missing_required == ["nlp"]
    assert result.rag_rejected["nlp"]["indicators"] == ["text classification"]


@pytest.mark.asyncio
async def test_one_mention_is_never_counted_as_two_indicators() -> None:
    # "faiss" is both a literal indicator and an alias of "vector database".
    result = await _enrich(_engine(DEVOPS), "devops", _candidate("Indexed embeddings with FAISS for search."))
    assert result.rag_enriched_skills == []
    assert len(result.rag_rejected["devops"]["indicators"]) == 1


@pytest.mark.asyncio
async def test_not_sufficient_phrase_masks_the_indicator_it_covers() -> None:
    candidate = _candidate("Implemented payment tokenization for cards and ran customer sentiment analysis surveys.")
    result = await _enrich(_engine(NLP), "nlp", candidate)
    assert result.rag_enriched_skills == []
    assert "nlp" not in result.rag_rejected  # no indicator survived masking


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [
    "No experience with Terraform or Jenkins; deployments were manual.",
    "Currently learning Terraform and Jenkins through an online course.",
    "I want to learn Terraform and Jenkins next year.",
])
async def test_negated_or_learning_indicators_do_not_count(text) -> None:
    result = await _enrich(_engine(DEVOPS), "devops", _candidate(text))
    assert result.rag_enriched_skills == []
    assert "devops" not in result.rag_rejected


@pytest.mark.asyncio
async def test_negation_in_another_clause_does_not_block_an_indicator() -> None:
    candidate = _candidate("Automated releases with Jenkins and Terraform, but I have no experience with Java.")
    result = await _enrich(_engine(DEVOPS), "devops", candidate)
    assert result.rag_enriched_skills == ["devops"]


@pytest.mark.asyncio
async def test_explicitly_negated_skill_never_gets_credit() -> None:
    candidate = _candidate("Built BERT classifiers and tokenization pipelines, but I have no experience with NLP.")
    result = await _enrich(_engine(NLP), "nlp", candidate)
    assert result.rag_enriched_skills == []
    assert "negated" in result.rag_rejected["nlp"]["reason"]


@pytest.mark.asyncio
async def test_structured_learning_skill_blocks_its_indicator() -> None:
    candidate = _candidate("Worked with Jenkins and Terraform.", learning_skills=["terraform"])
    result = await _enrich(_engine(DEVOPS), "devops", candidate)
    assert result.rag_enriched_skills == []
    assert result.rag_rejected["devops"]["indicators"] == ["jenkins"]


@pytest.mark.asyncio
@pytest.mark.parametrize("similarity, expected", [(0.34, True), (0.50, True), (0.3399, False), (None, False)])
async def test_single_indicator_threshold_rule(monkeypatch, similarity, expected) -> None:
    monkeypatch.setattr(settings, "rag_chunk_similarity_threshold", 0.34)
    engine = _engine(NLP)

    async def fake_best_chunk(entry, evidence, chunk_indices):
        return evidence.chunks[chunk_indices[0]], similarity

    monkeypatch.setattr(engine, "_rag_best_chunk", fake_best_chunk)
    result = await _enrich(engine, "nlp", _candidate("Built a text classification service for emails."))
    assert (result.rag_enriched_skills == ["nlp"]) is expected
    if expected:
        assert result.rag_evidence["nlp"]["similarity"] == pytest.approx(similarity)
        assert result.rag_evidence["nlp"]["rule"].startswith("1 indicator")


class _StubEmbedder:
    """Maps the definition and chunks mentioning 'deployment' to one axis, everything else to another."""

    async def embed(self, texts):
        vectors = []
        for text in texts:
            on_topic = text == DEVOPS.definition or "deployment" in text.lower()
            vectors.append([1.0, 0.0] + [0.0] * 382 if on_topic else [0.0, 1.0] + [0.0] * 382)
        return vectors


@pytest.mark.asyncio
async def test_semantic_evidence_comes_from_the_indicator_chunk_only(monkeypatch) -> None:
    monkeypatch.setattr(settings, "embedding_provider", "sentence-transformers")
    monkeypatch.setattr(settings, "rag_chunk_similarity_threshold", 0.34)
    engine = _engine(DEVOPS)
    engine.embedder = _StubEmbedder()

    # An unrelated line that happens to match the definition must not rescue a lone indicator.
    off_topic = _candidate("Used Jenkins to email a nightly report.\nDocumented deployment steps for the team.")
    result = await _enrich(engine, "devops", off_topic)
    assert result.rag_enriched_skills == []
    assert result.rag_rejected["devops"]["similarity"] == pytest.approx(0.0)

    on_topic = _candidate("Automated deployment of services with Jenkins.\nEnjoys hiking.")
    result = await _enrich(engine, "devops", on_topic)
    assert result.rag_enriched_skills == ["devops"]
    assert result.rag_evidence["devops"]["best_chunk"] == "Automated deployment of services with Jenkins."
    assert result.rag_evidence["devops"]["similarity"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_rag_enrichment_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setattr(settings, "rag_enrichment_enabled", False)
    candidate = _candidate("Fine-tuned BERT models and wrote a custom tokenization pipeline.")
    result = await _enrich(_engine(NLP), "nlp", candidate)
    assert result.missing_required == ["nlp"]


@pytest.mark.asyncio
async def test_match_reasoning_explains_rag_credit() -> None:
    await init_db()
    async with SessionLocal() as session:
        await sync_knowledge_base(session)
        job = Job(id=str(uuid.uuid4()), title="NLP Engineer", description="Language models for support.",
                  required_skills=["nlp", "python"], optional_skills=[], seniority="mid")
        candidate = _candidate(
            "Python developer. Fine-tuned RoBERTa models and built named entity recognition for invoices.",
            skills=["python"],
        )
        result = await HybridMatchingEngine()._compute_match(job, candidate, semantic_score=0.0, rag_session=session)

    assert result is not None
    evidence = result.reasoning.score_trace["rag_evidence"]["nlp"]
    assert set(evidence["indicators"]) == {"roberta", "named entity recognition"}
    assert evidence["best_chunk"]
    assert "similarity" in evidence
    payload = result.to_dict()
    assert payload["rag_evidence"]["nlp"]["indicators"] == evidence["indicators"]
    assert payload["rag_enriched_skills"] == ["nlp"]
    assert any("nlp" in strength for strength in payload["strengths"])
    rag_match = next(match for match in result.skill_match.matched_required if match.skill == "nlp")
    assert rag_match.confidence <= 0.60
