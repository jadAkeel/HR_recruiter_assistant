from app.services import skill_status as ss
from app.services.cv_parser import parse_cv_text


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
