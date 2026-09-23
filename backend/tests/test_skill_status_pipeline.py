import asyncio
import uuid

import pytest
from sqlalchemy import select

from app.core.db import SessionLocal, init_db
from app.models.candidate import Candidate
from app.models.job import Job
from app.models.match_result import MatchResult
from app.services import skill_status as ss
from app.services import skill_status_refinement as refinement
from app.services.cv_parser import parse_cv_text
from app.services.hybrid_matcher import HybridMatchingEngine


# --- Parsers store rule statuses ----------------------------------------------------

def test_default_parser_stores_rule_statuses() -> None:
    """
    Checks that the default upload path stores status, source and evidence.
    """
    profile = parse_cv_text(
        "Jane Doe\nSummary\nI know Python but not JavaScript. I am learning Go.\n"
        "Experience\nBuilt data pipelines with SQL at Acme. Basic knowledge of Docker."
    )
    details = {detail.name: detail for detail in profile.skills_detailed}
    assert details["javascript"].status.value == "no_experience"
    assert details["go"].status.value == "learning"
    assert details["docker"].status.value == "weak"
    assert details["sql"].status.value == "has_experience"
    assert details["sql"].source == ss.SOURCE_RULE
    assert details["sql"].context == "Built data pipelines with SQL at Acme."
    assert "javascript" not in profile.skills and profile.negative_skills == ["javascript"]
    assert "go" in profile.skills and profile.learning_skills == ["go"]


def test_parser_handles_skills_whose_catalog_name_normalizes_differently() -> None:
    """
    Regression: statuses are keyed by normalized name, so aliases such as "milvus"
    (normalized to "vector database") must not crash the parser.
    """
    profile = parse_cv_text("Jane Doe\nExperience\nBuilt a semantic search service on Milvus and Python.")
    assert "python" in profile.skills
    assert any(detail.name == "vector database" for detail in profile.skills_detailed)


# --- Matcher reads stored statuses ---------------------------------------------------

def _candidate_with(details_text: str, statuses: dict[str, str]) -> Candidate:
    """
    Builds a candidate whose stored statuses were decided by the LLM for sentences no
    rule would resolve, to prove the matcher reads storage instead of re-deriving.
    """
    results = ss.assess_cv_text(details_text, list(statuses))
    for skill, label in statuses.items():
        for item in results[skill].evidence:
            item.status, item.source, item.matched, item.llm_checked = label, ss.SOURCE_LLM, True, True
        results[skill] = ss.aggregate(skill, results[skill].evidence)
    fields = ss.to_profile_fields(results)
    return Candidate(
        id=str(uuid.uuid4()), full_name="Stored", email=f"{uuid.uuid4().hex[:6]}@example.com", phone=None,
        skills=[s for s in statuses if statuses[s] != ss.NONE], skills_detailed=fields["skills_detailed"],
        negative_skills=fields["negative_skills"], learning_skills=fields["learning_skills"],
        experience=[], education=[], projects=[], raw_text=details_text,
    )


@pytest.mark.asyncio
async def test_matcher_applies_stored_statuses_and_explains_them() -> None:
    """
    Checks NONE -> rejected, LEARNING -> 0.60, WEAK -> 0.80, HAS -> 1.0, all from
    stored statuses, with status/source/evidence in the score trace.
    """
    text = "Experience\nPython at Acme.\nDocker at Acme.\nKubernetes at Acme.\nJavaScript at Acme."
    candidate = _candidate_with(text, {"python": ss.HAS, "docker": ss.WEAK, "kubernetes": ss.LEARNING, "javascript": ss.NONE})
    job = Job(
        id=str(uuid.uuid4()), title="Platform", description="Platform engineer.",
        required_skills=["python", "docker", "kubernetes", "javascript"], optional_skills=[], seniority="mid",
    )

    result = await HybridMatchingEngine()._compute_match(job, candidate, semantic_score=0.0)

    confidences = {match.skill: match.confidence for match in result.skill_match.matched_required}
    assert confidences == {"python": 1.0, "docker": 0.80, "kubernetes": 0.60}
    assert result.skill_match.missing_required == ["javascript"]
    trace = {entry["skill"]: entry for entry in result.reasoning.score_trace["skill_statuses"]}
    assert trace["javascript"]["status"] == ss.NONE and trace["javascript"]["evidence"] == "JavaScript at Acme."
    assert trace["docker"]["status_source"] == ss.SOURCE_LLM and trace["docker"]["confidence_cap"] == 0.80


# --- Background refinement: fallback and idempotency --------------------------------

async def _saved_candidate_and_match(text: str) -> tuple[str, str]:
    await init_db()
    profile = parse_cv_text(text)
    candidate_id, job_id = str(uuid.uuid4()), str(uuid.uuid4())
    async with SessionLocal() as session:
        session.add(Candidate(
            id=candidate_id, full_name=profile.full_name, email=f"{candidate_id[:8]}@example.com", phone=None,
            skills=profile.skills, skills_detailed=[d.model_dump() for d in profile.skills_detailed],
            negative_skills=profile.negative_skills, learning_skills=profile.learning_skills,
            experience=profile.experience, education=profile.education, projects=profile.projects,
            raw_text=profile.raw_text,
        ))
        session.add(Job(
            id=job_id, title="Frontend", description="Frontend role.",
            required_skills=["javascript", "python"], optional_skills=[], seniority="mid",
        ))
        session.add(MatchResult(job_id=job_id, candidate_id=candidate_id, score=0.9, reasoning={}, is_stale=False))
        await session.commit()
    return candidate_id, job_id


@pytest.mark.asyncio
async def test_refinement_keeps_rules_when_ollama_is_down(monkeypatch) -> None:
    """
    Checks that a failing LLM leaves the stored rule results untouched.
    """
    candidate_id, _ = await _saved_candidate_and_match("Jane\nSummary\nJavaScript and TypeScript.\nI am learning Go.")
    monkeypatch.setattr(refinement.settings, "skill_status_llm_enabled", True)

    async def down(items, timeout=None):
        raise ss.LLMUnavailable("ConnectError: refused")

    monkeypatch.setattr(ss, "llm_classify", down)
    async with SessionLocal() as session:
        before = (await session.get(Candidate, candidate_id)).skills_detailed
        summary = await refinement.refine_candidate_skill_status(session, candidate_id)
    async with SessionLocal() as session:
        after = (await session.get(Candidate, candidate_id)).skills_detailed

    assert summary["status"] == "llm_unavailable"
    assert after == before


@pytest.mark.asyncio
async def test_refinement_is_idempotent_and_marks_matches_stale(monkeypatch) -> None:
    """
    Checks that refinement stores LLM labels, marks saved matches stale when a status
    changes, and that a second run does not call the LLM or change anything.
    """
    candidate_id, _ = await _saved_candidate_and_match(
        "Jane\nSummary\nJavaScript and TypeScript.\nI am learning Go.\nExperience\nDeveloped APIs in Python at Acme."
    )
    monkeypatch.setattr(refinement.settings, "skill_status_llm_enabled", True)
    calls: list[list[tuple[str, str]]] = []

    async def fake_llm(items, timeout=None):
        calls.append(items)
        return {index: ("WEAK" if skill == "javascript" else "HAS") for index, (skill, _) in enumerate(items, start=1)}, 0.1

    monkeypatch.setattr(ss, "llm_classify", fake_llm)
    async with SessionLocal() as session:
        first = await refinement.refine_candidate_skill_status(session, candidate_id, ss.raw_text_hash(
            (await session.get(Candidate, candidate_id)).raw_text
        ))
    async with SessionLocal() as session:
        candidate = await session.get(Candidate, candidate_id)
        details_after_first = candidate.skills_detailed
        match = (await session.execute(select(MatchResult).where(MatchResult.candidate_id == candidate_id))).scalar_one()
        second = await refinement.refine_candidate_skill_status(session, candidate_id)
    async with SessionLocal() as session:
        details_after_second = (await session.get(Candidate, candidate_id)).skills_detailed

    assert first["status"] == "refined" and "javascript" in first["changed_skills"]
    assert {skill for skill, _ in calls[0]} == {"javascript", "typescript"}  # Go and Python were rule-resolved
    stored = {detail["name"]: detail for detail in details_after_first}
    assert (stored["javascript"]["status"], stored["javascript"]["source"]) == ("weak", ss.SOURCE_LLM)
    assert match.is_stale is True
    assert second["status"] == "noop" and len(calls) == 1
    assert details_after_second == details_after_first


@pytest.mark.asyncio
async def test_refinement_skips_a_cv_replaced_after_queueing(monkeypatch) -> None:
    """
    Checks that a task for an older version of the CV does nothing.
    """
    candidate_id, _ = await _saved_candidate_and_match("Jane\nSummary\nJavaScript and TypeScript.")
    monkeypatch.setattr(refinement.settings, "skill_status_llm_enabled", True)
    async with SessionLocal() as session:
        summary = await refinement.refine_candidate_skill_status(session, candidate_id, "old-hash")
    assert summary["status"] == "cv_changed"


@pytest.mark.asyncio
async def test_upload_queues_refinement_only_when_items_are_unresolved(monkeypatch) -> None:
    """
    Checks the upload hook: it queues a task for unresolved items and never raises.
    """
    from app.services import task_queue

    monkeypatch.setattr(refinement.settings, "skill_status_llm_enabled", True)
    queued = {}

    async def fake_enqueue(**kwargs):
        queued.update(kwargs)
        return "task-1"

    monkeypatch.setattr(task_queue, "enqueue_skill_status_refinement", fake_enqueue)
    profile = parse_cv_text("Jane\nSummary\nJavaScript and TypeScript.")
    candidate = Candidate(id="c-1", raw_text=profile.raw_text, skills_detailed=[d.model_dump() for d in profile.skills_detailed])
    assert await refinement.enqueue_if_needed(candidate) == "task-1"
    assert queued["candidate_id"] == "c-1"

    resolved = parse_cv_text("Jane\nSummary\nI am learning Go.")
    candidate = Candidate(id="c-2", raw_text=resolved.raw_text, skills_detailed=[d.model_dump() for d in resolved.skills_detailed])
    assert await refinement.enqueue_if_needed(candidate) is None

    async def broken_enqueue(**kwargs):
        raise RuntimeError("Redis is required")

    monkeypatch.setattr(task_queue, "enqueue_skill_status_refinement", broken_enqueue)
    candidate = Candidate(id="c-3", raw_text=profile.raw_text, skills_detailed=[d.model_dump() for d in profile.skills_detailed])
    assert await refinement.enqueue_if_needed(candidate) is None


@pytest.mark.asyncio
async def test_upload_hook_does_not_wait_for_the_queue(monkeypatch) -> None:
    """
    Checks that the API upload path schedules the enqueue in the background, so a slow
    or unreachable Redis cannot delay the upload response.
    """
    import time

    from app.services import task_queue

    monkeypatch.setattr(refinement.settings, "skill_status_llm_enabled", True)
    queued = asyncio.Event()

    async def slow_enqueue(**kwargs):
        await asyncio.sleep(0.5)
        queued.set()
        return "task-slow"

    monkeypatch.setattr(task_queue, "enqueue_skill_status_refinement", slow_enqueue)
    profile = parse_cv_text("Jane\nSummary\nJavaScript and TypeScript.")
    candidate = Candidate(id="c-4", raw_text=profile.raw_text, skills_detailed=[d.model_dump() for d in profile.skills_detailed])

    started = time.monotonic()
    assert refinement.schedule_refinement(candidate) is True
    assert time.monotonic() - started < 0.1
    await asyncio.wait_for(queued.wait(), timeout=5)


@pytest.mark.asyncio
async def test_stale_saved_match_is_rescored_on_read() -> None:
    """
    Checks that a match marked stale is recomputed even if its scoring model is current.
    """
    from app.api.matching import _refresh_stale_saved_matches

    candidate_id, job_id = await _saved_candidate_and_match(
        "Jane\nSummary\nI know Python but not JavaScript.\nExperience\nDeveloped APIs in Python at Acme."
    )
    engine = HybridMatchingEngine()
    async with SessionLocal() as session:
        job = await session.get(Job, job_id)
        candidate = await session.get(Candidate, candidate_id)
        current = await engine._compute_match(job, candidate, semantic_score=0.0)
        match = (await session.execute(select(MatchResult).where(MatchResult.candidate_id == candidate_id))).scalar_one()
        match.reasoning = current.to_dict()
        match.score = 0.99
        match.is_stale = True
        await session.commit()
        refreshed = await _refresh_stale_saved_matches(session, job, [match])

    assert refreshed[0].is_stale is False
    assert refreshed[0].score == pytest.approx(current.final_score)


@pytest.mark.asyncio
async def test_refinement_enqueue_falls_back_to_memory_when_redis_push_fails(monkeypatch) -> None:
    """
    Checks that a refinement task is not lost when Redis fails mid-push (development).
    """
    from app.services import task_queue

    class FailingRedis:
        async def lpush(self, *args):
            raise ConnectionError("refused")

    async def failing_redis():
        return FailingRedis()

    monkeypatch.setattr(task_queue, "get_redis", failing_redis)
    task_id = await task_queue.enqueue_skill_status_refinement("cand-9", "hash")
    task = task_queue._in_memory_tasks.pop(task_id)
    assert task["type"] == task_queue.SKILL_STATUS_TASK and task["candidate_id"] == "cand-9"
