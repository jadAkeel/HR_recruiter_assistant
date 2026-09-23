import asyncio
import json
from pathlib import Path

import pytest

from app.services import skill_status as ss

FIXTURE = Path(__file__).parent / "fixtures" / "skill_status_sentences.jsonl"

# Measured when the rules were frozen (scripts/benchmark_skill_status.py):
# seen 100%, held-out 55%. The floors leave a little room and catch regressions.
SEEN_ACCURACY_FLOOR = 0.95
HELDOUT_ACCURACY_FLOOR = 0.50


def _fixture(split: str | None = None) -> list[dict]:
    items = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [item for item in items if split in (None, item["split"])]


def _rule_label(item: dict) -> str:
    return ss.classify_rules(item["skill"], item["sentence"]).status


# --- Rule layer: accuracy gates (no LLM, runs in CI) -------------------------------

def test_rule_accuracy_on_seen_set_meets_floor() -> None:
    """
    Fails when a rule change regresses the sentences the rules were written from.
    """
    items = _fixture("seen")
    correct = sum(_rule_label(item) == item["label"] for item in items)
    assert correct / len(items) >= SEEN_ACCURACY_FLOOR, f"{correct}/{len(items)}"


def test_rule_accuracy_on_heldout_set_meets_floor() -> None:
    """
    Fails when a rule change makes the rules generalise worse than when frozen.
    """
    items = _fixture("heldout")
    correct = sum(_rule_label(item) == item["label"] for item in items)
    assert correct / len(items) >= HELDOUT_ACCURACY_FLOOR, f"{correct}/{len(items)}"


def test_rules_never_reject_a_real_has_skill() -> None:
    """
    HAS -> NONE is the most harmful error: it removes a skill the candidate has.
    """
    wrong = [item["sentence"] for item in _fixture() if item["label"] == ss.HAS and _rule_label(item) == ss.NONE]
    assert wrong == []


# --- Rule layer: behaviour -----------------------------------------------------------

@pytest.mark.parametrize(
    "sentence",
    [
        "I don't know JavaScript.",
        "I don’t know JavaScript.",
        "I dont know JavaScript.",
        "I do not know JavaScript.",
        "I haven't used JavaScript.",
        "I havent used JavaScript.",
    ],
)
def test_apostrophe_variants_are_negations(sentence: str) -> None:
    """
    Checks that don't / don’t / dont / do not all read as NONE.
    """
    assessment = ss.classify_rules("javascript", sentence)
    assert (assessment.status, assessment.matched, assessment.source) == (ss.NONE, True, ss.SOURCE_RULE)


def test_negation_is_scoped_to_the_named_skill() -> None:
    """
    Checks that "Python but not JavaScript" only negates JavaScript.
    """
    sentence = "I know Python but not JavaScript."
    assert ss.classify_rules("javascript", sentence).status == ss.NONE
    assert ss.classify_rules("python", sentence).status != ss.NONE

    sentence = "Python developer; I have never touched JavaScript."
    assert ss.classify_rules("javascript", sentence).status == ss.NONE
    assert ss.classify_rules("python", sentence).status == ss.HAS


def test_rule_order_and_unresolved_default() -> None:
    """
    Checks NONE before LEARNING before WEAK, and that no rule means unresolved HAS.
    """
    assert ss.classify_rules("javascript", "Not familiar with JavaScript.").status == ss.NONE
    assert ss.classify_rules("javascript", "Currently learning JavaScript.").status == ss.LEARNING
    assert ss.classify_rules("javascript", "Basic knowledge of JavaScript.").status == ss.WEAK
    explicit = ss.classify_rules("javascript", "Built a React dashboard in JavaScript.")
    assert (explicit.status, explicit.matched) == (ss.HAS, True)
    unresolved = ss.classify_rules("javascript", "JavaScript and a bit of everything else.")
    assert (unresolved.status, unresolved.matched, unresolved.source) == (ss.HAS, False, ss.SOURCE_DEFAULT)


def test_skill_names_containing_learning_are_not_learning_status() -> None:
    """
    Checks that "machine learning" / "deep learning" do not mark a skill as LEARNING.
    """
    assert ss.classify_rules("python", "Built deep learning models in Python.").status == ss.HAS
    assert ss.classify_rules("python", "Python for machine learning pipelines.").status != ss.LEARNING


def test_only_sentences_naming_the_skill_are_assessed() -> None:
    """
    Checks sentence selection: a negation about another skill does not leak.
    """
    text = "Summary\nI have no experience with Docker.\nExperience\nDeveloped REST APIs in Python at Acme."
    results = ss.assess_cv_text(text, ["python", "docker"])
    assert results["docker"].status == ss.NONE
    assert results["python"].status == ss.HAS
    assert [item.evidence_sentence for item in results["python"].evidence] == ["Developed REST APIs in Python at Acme."]


def test_experience_evidence_beats_summary_claims() -> None:
    """
    Checks that HAS evidence in Experience/Projects wins over a limiting Summary line,
    and that an explicit limitation beats a bare Skills-list mention.
    """
    text = (
        "Summary\nI am learning Docker.\n"
        "Experience\nDeveloped container images with Docker for 12 services.\n"
        "Skills: JavaScript\nBasic knowledge of JavaScript."
    )
    results = ss.assess_cv_text(text, ["docker", "javascript"])
    assert results["docker"].status == ss.HAS
    assert results["docker"].section == "experience"
    assert results["javascript"].status == ss.WEAK


def test_bullet_starting_with_a_header_word_is_content() -> None:
    """
    Checks that "Experience with X." is a sentence, not a section header.
    """
    sentences = ss.split_cv_sentences("Work Experience\n- Experience with Kubernetes.\nSkills: Python, SQL")
    assert ("Experience with Kubernetes.", "experience") in sentences
    assert ("Skills: Python, SQL", "skills") in sentences


# --- LLM layer: output validation (mocked, no network) -----------------------------

def test_llm_output_is_validated_strictly() -> None:
    """
    Checks that only a JSON object of in-range item numbers to known labels is used.
    """
    assert ss.parse_llm_labels('{"1": "HAS", "2": "weak", "3": "MAYBE", "9": "NONE"}', 3) == {1: "HAS", 2: "WEAK"}
    assert ss.parse_llm_labels('["HAS", "NONE"]', 2) == {}
    assert ss.parse_llm_labels("not json", 2) == {}
    assert ss.parse_llm_labels('{"1.": "LEARNING", "x": "HAS", "2": 3}', 2) == {1: "LEARNING"}


def test_llm_none_needs_a_negation_in_the_sentence() -> None:
    """
    Checks the guard against the LLM rejecting obvious HAS sentences.
    """
    assert not ss.accept_llm_label(ss.NONE, "JavaScript and TypeScript on every project.")
    assert ss.accept_llm_label(ss.NONE, "Have yet to use JavaScript.")
    assert ss.accept_llm_label(ss.WEAK, "JavaScript and TypeScript on every project.")


def test_refine_sends_only_unresolved_items_and_keeps_rules_on_bad_labels(monkeypatch) -> None:
    """
    Checks that rule-resolved sentences never reach the LLM, and an unknown label keeps
    the rule result.
    """
    text = "Summary\nI am learning Go.\nJavaScript and TypeScript.\nPython scripts at home."
    results = ss.assess_cv_text(text, ["go", "javascript", "typescript", "python"])
    sent: list[tuple[str, str]] = []

    async def fake_llm(items, timeout=None):
        sent.extend(items)
        labels = {"javascript": "WEAK", "typescript": "MAYBE", "python": "LEARNING"}
        return {index: labels[skill] for index, (skill, _) in enumerate(items, start=1) if labels[skill] in ss.LABELS}, 0.1

    monkeypatch.setattr(ss, "llm_classify", fake_llm)
    summary = asyncio.run(ss.refine_results(results))

    assert {skill for skill, _ in sent} == {"javascript", "typescript", "python"}
    assert summary["status"] == "refined"
    assert (results["javascript"].status, results["javascript"].source) == (ss.WEAK, ss.SOURCE_LLM)
    assert (results["typescript"].status, results["typescript"].source) == (ss.HAS, ss.SOURCE_DEFAULT)
    assert (results["go"].status, results["go"].source) == (ss.LEARNING, ss.SOURCE_RULE)
    assert ss.unresolved_items(results) == []


def test_llm_unreachable_raises_llm_unavailable(monkeypatch) -> None:
    """
    Checks that a down Ollama surfaces as LLMUnavailable (callers keep rule results).
    """
    monkeypatch.setattr(ss.settings, "ollama_base_url", "http://127.0.0.1:9")
    with pytest.raises(ss.LLMUnavailable):
        asyncio.run(ss.llm_classify([("javascript", "JavaScript and TypeScript.")], timeout=2))


