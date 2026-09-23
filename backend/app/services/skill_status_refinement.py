"""
Background LLM refinement of a candidate's stored skill statuses.

Uploads store rule-based statuses immediately (app.services.skill_status). This task
then sends only the unresolved (skill, sentence) items to the local LLM, saves the
result, and marks the candidate's saved matches stale so they are re-scored.

Idempotent: items the LLM has already checked are skipped, so a second run is a no-op.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.candidate import Candidate
from app.models.match_result import MatchResult
from app.services import skill_status
from app.services.skill_catalog import normalize_skill_name

logger = logging.getLogger(__name__)

# Strong references to fire-and-forget enqueue tasks, so they are not garbage-collected.
_pending_enqueues: set[asyncio.Task] = set()

# SkillDetail fields that come from the parser, not from the status layer.
_PRESERVED_DETAIL_KEYS = ("level", "years", "esco_uri", "preferred_label", "category")


def has_unresolved_items(skills_detailed: list[Any] | None) -> bool:
    """
    Checks whether stored statuses still contain items for the LLM.
    """
    details = [
        detail.model_dump(mode="json") if hasattr(detail, "model_dump") else detail
        for detail in skills_detailed or []
    ]
    return bool(skill_status.unresolved_items(skill_status.results_from_details(details)))


async def enqueue_if_needed(candidate: Candidate) -> str | None:
    """
    Queues refinement for a freshly saved candidate when some items are unresolved.
    Never raises: an upload must not fail because the queue is unavailable.
    """
    if not settings.skill_status_llm_enabled or not has_unresolved_items(candidate.skills_detailed):
        return None
    try:
        from app.services.task_queue import enqueue_skill_status_refinement

        return await enqueue_skill_status_refinement(
            candidate_id=candidate.id,
            raw_text_hash=skill_status.raw_text_hash(candidate.raw_text),
            created_by_user_id=candidate.created_by_user_id,
        )
    except Exception:
        logger.warning("Could not queue skill status refinement", extra={"candidate_id": candidate.id}, exc_info=True)
        return None


def schedule_refinement(candidate: Candidate) -> bool:
    """
    Queues refinement without making the caller wait, even for the queue itself: with
    Redis down, each connection attempt can take seconds, and uploads must stay fast.
    Returns True when a task was scheduled.
    """
    if not settings.skill_status_llm_enabled or not has_unresolved_items(candidate.skills_detailed):
        return False
    snapshot = Candidate(
        id=candidate.id,
        raw_text=candidate.raw_text,
        created_by_user_id=candidate.created_by_user_id,
        skills_detailed=candidate.skills_detailed,
    )
    task = asyncio.get_running_loop().create_task(enqueue_if_needed(snapshot))
    _pending_enqueues.add(task)
    task.add_done_callback(_pending_enqueues.discard)
    return True


def _merged_details(old_details: list[Any], fields: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Keeps parser-owned keys (years, level, ESCO data) when statuses are rewritten.
    """
    previous = {
        normalize_skill_name(str(detail.get("name") or "")): detail
        for detail in old_details or []
        if isinstance(detail, dict)
    }
    merged = []
    for detail in fields["skills_detailed"]:
        old = previous.get(detail["name"], {})
        merged.append({**{key: old[key] for key in _PRESERVED_DETAIL_KEYS if key in old}, **detail})
    return merged


async def refine_candidate_skill_status(
    session: AsyncSession,
    candidate_id: str,
    raw_text_hash: str | None = None,
) -> dict[str, Any]:
    """
    Runs the LLM on a candidate's unresolved skill sentences and stores the result.

    Returns a summary whose "status" is one of: missing, cv_changed, disabled, noop,
    llm_unavailable, refined.
    """
    candidate = await session.get(Candidate, candidate_id)
    if candidate is None:
        return {"status": "missing", "candidate_id": candidate_id}
    if raw_text_hash and skill_status.raw_text_hash(candidate.raw_text) != raw_text_hash:
        # The CV was replaced after this task was queued; the new upload queues its own.
        return {"status": "cv_changed", "candidate_id": candidate_id}
    if not settings.skill_status_llm_enabled:
        return {"status": "disabled", "candidate_id": candidate_id}

    results = skill_status.results_from_details(candidate.skills_detailed or [])
    if not results:
        skills = list(candidate.skills or []) + list(candidate.negative_skills or [])
        results = skill_status.assess_cv_text(candidate.raw_text or "", skills)
    before = {skill: result.status for skill, result in results.items()}

    try:
        summary = await skill_status.refine_results(results)
    except skill_status.LLMUnavailable as exc:
        logger.warning("Skill status LLM unavailable; rule results kept", extra={"candidate_id": candidate_id, "error": str(exc)[:200]})
        return {"status": "llm_unavailable", "candidate_id": candidate_id}
    if summary["status"] == "noop":
        return {**summary, "candidate_id": candidate_id}

    fields = skill_status.to_profile_fields(results)
    candidate.skills_detailed = _merged_details(candidate.skills_detailed or [], fields)
    candidate.learning_skills = fields["learning_skills"]
    candidate.negative_skills = fields["negative_skills"]
    negative = set(fields["negative_skills"])
    kept = [skill for skill in candidate.skills or [] if normalize_skill_name(skill) not in negative]
    known = {normalize_skill_name(skill) for skill in kept}
    candidate.skills = kept + [skill for skill in results if skill not in negative and skill not in known]

    changed = sorted(skill for skill, result in results.items() if before.get(skill) != result.status)
    stale = 0
    if changed:
        outcome = await session.execute(
            update(MatchResult).where(MatchResult.candidate_id == candidate_id).values(is_stale=True)
        )
        stale = outcome.rowcount or 0
    await session.commit()

    from app.services.skill_evidence import replace_candidate_skill_evidence

    await replace_candidate_skill_evidence(session, candidate, commit=True)
    logger.info("Skill statuses refined", extra={"candidate_id": candidate_id, "changed": changed, "stale_matches": stale})
    return {**summary, "candidate_id": candidate_id, "changed_skills": changed, "stale_matches": stale}
