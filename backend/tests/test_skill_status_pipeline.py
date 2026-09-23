import uuid

import pytest

from app.models.candidate import Candidate
from app.models.job import Job
from app.services import skill_status as ss
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
