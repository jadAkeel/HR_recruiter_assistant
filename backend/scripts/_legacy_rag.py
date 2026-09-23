"""
Faithful copy of the pre-2026-09 RAG enrichment, kept only for benchmarks.

OLD rule: embed the skill definition and the WHOLE candidate text, grant a
0.60-confidence "rag" match when cosine > 0.75 (token-overlap fallback when
embeddings are unavailable).
"""
from __future__ import annotations

import re
from typing import Any

from app.core.config import settings
from app.services.embedding import is_embedding_quality_text
from app.services.hybrid_matcher import HybridMatchingEngine, SkillMatch, SkillMatchResult
from app.services.rag import INITIAL_KNOWLEDGE, get_skill_definitions
from app.services.skill_catalog import normalize_skill_name

LEGACY_THRESHOLD = 0.75


def legacy_initial_definitions(skills: list[str]) -> dict[str, str]:
    """
    Old get_skill_definitions scoring over the 5 original INITIAL_KNOWLEDGE docs.
    """
    documents = [doc for doc in INITIAL_KNOWLEDGE if doc["category"] == "skill"]
    definitions: dict[str, str] = {}
    for skill in (" ".join(s.lower().split()) for s in skills):
        best: tuple[dict[str, Any], int] | None = None
        for document in documents:
            title = " ".join(document["title"].lower().split())
            tags = {" ".join(str(tag).lower().split()) for tag in document["tags"]}
            score = 3 if skill in tags else 2 if skill in title else 1 if skill in document["content"].lower() else 0
            if score and (best is None or score > best[1]):
                best = (document, score)
        if best:
            definitions[skill] = best[0]["content"]
    return definitions


def _content_tokens(text: str) -> set[str]:
    stopwords = {
        "and", "are", "for", "from", "into", "that", "the", "this", "with",
        "skill", "skills", "using", "used", "use", "key", "known",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]{2,}", (text or "").lower())
        if token not in stopwords
    }


async def legacy_similarity(engine: HybridMatchingEngine, definition: str, candidate: Any) -> float:
    """
    Old _rag_similarity_for_skill: cosine between definition and the whole CV text.
    """
    candidate_text = engine._build_candidate_text(candidate)
    if not is_embedding_quality_text(definition) or not is_embedding_quality_text(candidate_text):
        return 0.0
    if settings.embedding_provider.lower() != "hash":
        try:
            import numpy as np

            left, right = await engine.embedder.embed([definition, candidate_text])
            if len(left) == len(right):
                a = np.array(left, dtype=np.float32)
                b = np.array(right, dtype=np.float32)
                denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                if denom:
                    return float(np.dot(a, b) / denom)
        except Exception:
            pass
    definition_tokens = _content_tokens(definition)
    candidate_tokens = _content_tokens(candidate_text)
    if not definition_tokens or not candidate_tokens:
        return 0.0
    return min(1.0, len(definition_tokens & candidate_tokens) / len(definition_tokens) * 2.0)


class LegacyRagEngine(HybridMatchingEngine):
    """
    HybridMatchingEngine with the old RAG enrichment, for before/after comparisons.
    """

    async def _apply_rag_enrichment(self, result: SkillMatchResult, candidate: Any, session: Any) -> None:
        try:
            definitions = await get_skill_definitions(result.missing_required, session=session)
        except Exception:
            return
        still_missing: list[str] = []
        for skill in result.missing_required:
            definition = definitions.get(normalize_skill_name(skill))
            if not definition:
                still_missing.append(skill)
                continue
            similarity = await legacy_similarity(self, definition, candidate)
            if similarity > LEGACY_THRESHOLD:
                result.matched_required.append(SkillMatch(skill=skill, match_type="rag", confidence=0.60))
                result.rag_matched_count += 1
                result.rag_enriched_skills.append(skill)
                result.rag_evidence[skill] = {"similarity": round(similarity, 4), "rule": "legacy whole-CV > 0.75"}
            else:
                still_missing.append(skill)
        result.missing_required = still_missing
