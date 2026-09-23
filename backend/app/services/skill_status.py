"""
Decides, per skill, whether a CV shows the candidate HAS it, knows it only at a WEAK
level, is LEARNING it, or does NOT have it (NONE). English only.

Two layers, shared by the CV parsers and the matcher:
1. Rules (fast, at upload): each CV sentence that mentions a skill is checked, within
   the clause that names the skill, in the order NONE -> LEARNING -> WEAK -> explicit
   HAS evidence. A sentence no rule matches defaults to HAS but stays unresolved.
2. LLM (background): only unresolved (skill, sentence) items are sent to the local
   model in one batched call; its label replaces the default when it is valid.

The rule vocabulary is taken from the seen benchmark sentences
(tests/fixtures/skill_status_sentences.jsonl), generalised only by contractions,
inflections, and clause scoping, so held-out accuracy stays an honest measure.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from app.core.config import settings
from app.services.skill_catalog import build_skill_pattern, normalize_skill_name

logger = logging.getLogger(__name__)

HAS = "HAS"
WEAK = "WEAK"
LEARNING = "LEARNING"
NONE = "NONE"
LABELS = (HAS, WEAK, LEARNING, NONE)

SOURCE_RULE = "rule"
SOURCE_LLM = "llm"
SOURCE_DEFAULT = "default"  # no rule matched; HAS until the LLM says otherwise

# Values stored in SkillDetail.status (the first three predate this module).
STORED_STATUS = {
    HAS: "has_experience",
    WEAK: "weak",
    LEARNING: "learning",
    NONE: "no_experience",
}
LABEL_FROM_STORED = {value: label for label, value in STORED_STATUS.items()}

# Matching confidence caps per status; NONE rejects the skill.
STATUS_CONFIDENCE_CAP = {HAS: 1.0, WEAK: 0.80, LEARNING: 0.60, NONE: 0.0}

# How much a status is worth when several sentences disagree.
_POSITIVITY = {HAS: 3, WEAK: 2, LEARNING: 1, NONE: 0}
EVIDENCE_SECTIONS = {"experience", "projects"}

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_APOSTROPHES = re.compile(r"[’‘ʼ`´]")
_CONTRACTIONS = [
    (re.compile(r"\bcan'?t\b"), "can not"),
    (re.compile(r"\bwon'?t\b"), "will not"),
    (re.compile(r"\b(do|does|did|have|has|had|is|are|was|were|could|would|should)n'?t\b"), r"\1 not"),
    (re.compile(r"\bi'm\b"), "i am"),
    (re.compile(r"\bi'?ve\b"), "i have"),
]


def normalize_for_rules(text: str) -> str:
    """
    Lowercases and expands contractions, so don't / dont / don't / do not all match.
    """
    lowered = _APOSTROPHES.sub("'", str(text or "").lower())
    for pattern, replacement in _CONTRACTIONS:
        lowered = pattern.sub(replacement, lowered)
    return re.sub(r"\s+", " ", lowered).strip()


# Clause boundaries that change who/what a status word applies to:
# "I know Python but not JavaScript" -> ["i know python", "not javascript"].
_CLAUSE_SPLIT = re.compile(r";|\s+-\s+|\b(?:but|while|whereas|however|although|though|except)\b")

# ---------------------------------------------------------------------------
# Rules, checked in order within the clause that names the skill
# ---------------------------------------------------------------------------

_NONE_RULES = [
    # "no experience", "no prior javascript experience"
    re.compile(r"\bno\b(?:\s+[\w.+#-]+){0,3}?\s+(?:experience|knowledge|exposure)\b"),
    # "do not know", "never used", "have not worked", "never touched", "not familiar"
    re.compile(
        r"\b(?:not|never)\s+(?!only\b)(?:\w+\s+){0,2}?"
        r"(?:know|knew|known|use|used|using|work|worked|working|touch|touched|familiar|experienced)\b"
    ),
    # clause that starts with "not": the "not javascript" in "python but not javascript"
    re.compile(r"^(?:and\s+|or\s+)?not\b"),
]

# "learning" is also part of skill names; those are not learning statements.
_LEARNING_WORD = re.compile(
    r"(?<!machine )(?<!deep )(?<!reinforcement )(?<!transfer )(?<!supervised )(?<!e-)\blearning\b"
    r"(?!\s+(?:models?|algorithms?|pipelines?|engineer(?:ing)?|management|platforms?)\b)"
)
_LEARNING_RULES = [
    _LEARNING_WORD,                                   # "i am learning", "currently learning"
    re.compile(r"\bto\s+learn\b"),                    # "planning to learn"
    re.compile(r"(?<!case )\bstudy(?:ing)?\b|\bself[- ]study\b"),  # "studying ...", "self-study"
    re.compile(r"\btaking\s+(?:an?\s+|the\s+)?(?:[\w.+#-]+\s+){0,3}courses?\b"),  # "taking a javascript course"
    re.compile(r"\bteach(?:ing)?\s+myself\b"),        # "teaching myself"
    re.compile(r"\bimproving\s+my\b(?:\s+[\w.+#-]+){0,3}\s+skills?\b"),  # "improving my javascript skills"
]

_WEAK_RULES = [
    re.compile(r"\bbasics?\b"),                       # "basic knowledge"
    re.compile(r"\bfamiliar(?:ity)?\b"),              # "familiar with" ("not familiar" is NONE first)
    re.compile(r"\blimited\b"),                       # "limited experience"
    re.compile(r"\bnot\s+(?:very\s+)?good\b"),        # "not good at"
    re.compile(r"\bsome\s+exposure\b"),               # "some exposure"
    re.compile(r"\bbeginner\b"),                      # "beginner-level"
    re.compile(r"\belementary\b"),                    # "elementary understanding"
]

# Explicit positive evidence. Matching one resolves the item, so the LLM is not asked
# (it occasionally rejects obvious HAS sentences).
_HAS_RULES = [
    re.compile(r"\b(?:developed|built|worked)\b"),    # "developed rest apis", "built a dashboard"
    re.compile(r"\b\d+\+?\s*(?:years?|yrs?)\b"),      # "3 years of professional experience"
    re.compile(r"\b(?:expert|proficient|strong|hands-on|extensively)\b"),
    re.compile(r"\bknow\b.*\bwell\b"),                # "i know javascript well"
    re.compile(r"^(?:technical\s+)?skills?\s*:"),     # "skills: javascript, python"
]

# Generic English negation markers. An LLM "NONE" is accepted only for a sentence that
# contains one, since a claim of not having a skill needs a negation.
_NEGATION_CUE = re.compile(
    r"\b(?:not|no|never|none|nothing|without|neither|nor|zero|cannot|lack\w*|unfamiliar|unable|yet\s+to)\b"
)


def has_absence_or_learning_cue(text: str) -> bool:
    """
    Checks already-scoped text (e.g. the clause around a RAG indicator) for a NONE
    or LEARNING rule, so callers share this module's vocabulary.
    """
    normalized = normalize_for_rules(text)
    return any(rule.search(normalized) for rule in _NONE_RULES + _LEARNING_RULES)


@dataclass
class SkillAssessment:
    skill: str
    status: str = HAS
    matched: bool = False
    source: str = SOURCE_DEFAULT
    evidence_sentence: str | None = None
    section: str | None = None
    llm_checked: bool = False

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the assessment for storage in SkillDetail.evidence.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillAssessment":
        """
        Rebuilds an assessment from stored JSON, ignoring unknown keys.
        """
        known = {key: data[key] for key in cls.__dataclass_fields__ if key in data}
        assessment = cls(**known)
        if assessment.status not in LABELS:
            assessment.status = HAS
        return assessment


@dataclass
class SkillStatusResult:
    skill: str
    status: str
    source: str
    matched: bool
    evidence_sentence: str | None
    section: str | None
    evidence: list[SkillAssessment] = field(default_factory=list)


def _skill_clauses(normalized_sentence: str, skill: str) -> list[str]:
    """
    Returns the clauses of a sentence that name the skill.
    """
    pattern = build_skill_pattern(skill)
    return [
        clause.strip()
        for clause in _CLAUSE_SPLIT.split(normalized_sentence)
        if clause.strip() and pattern.search(clause)
    ]


def mentions_skill(sentence: str, skill: str) -> bool:
    """
    Checks whether a sentence names the skill or one of its aliases.
    """
    return bool(build_skill_pattern(skill).search(normalize_for_rules(sentence)))


def classify_rules(skill: str, sentence: str, section: str | None = None) -> SkillAssessment:
    """
    Classifies one sentence for one skill with the rule layer.

    Returns NONE, LEARNING, WEAK or explicit HAS with matched=True, or the HAS
    default with matched=False when no rule applies.
    """
    skill = normalize_skill_name(skill)
    normalized = normalize_for_rules(sentence)
    clauses = _skill_clauses(normalized, skill) or [normalized]
    scope = " | ".join(clauses)
    evidence = " ".join(str(sentence or "").split()) or None
    if section == "skills" and not any(rule.search(scope) for rule in _NONE_RULES + _LEARNING_RULES + _WEAK_RULES):
        return SkillAssessment(skill, HAS, True, SOURCE_RULE, evidence, section)

    for status, rules in ((NONE, _NONE_RULES), (LEARNING, _LEARNING_RULES), (WEAK, _WEAK_RULES), (HAS, _HAS_RULES)):
        for clause in clauses:
            if any(rule.search(clause) for rule in rules):
                return SkillAssessment(skill, status, True, SOURCE_RULE, evidence, section)
    return SkillAssessment(skill, HAS, False, SOURCE_DEFAULT, evidence, section)


# ---------------------------------------------------------------------------
# Sentence selection over a whole CV
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
_BULLET = re.compile(r"^[\s•●▪‣*\-–—>]+")


def _section_header(text: str) -> str | None:
    """
    Returns the section a short header text names ("Work Experience", "Skills").
    """
    from app.services.cv_parser import SECTION_HEADERS

    text = text.strip()
    if not text or len(text.split()) > 4 or not re.fullmatch(r"[A-Za-z &/()-]+", text):
        return None
    return next((name for name, pattern in SECTION_HEADERS.items() if pattern.search(text)), None)


def split_cv_sentences(text: str) -> list[tuple[str, str | None]]:
    """
    Splits CV text into (sentence, section) pairs, tracking the parser's section
    headers. "Skills: Python, SQL" both switches section and is kept as content;
    a bullet such as "Experience with Kubernetes." is content, not a header.
    """
    sentences: list[tuple[str, str | None]] = []
    section: str | None = None
    for raw_line in str(text or "").splitlines():
        line = _BULLET.sub("", raw_line).strip()
        if not line:
            continue
        header = _section_header(line.rstrip(":"))
        if header:
            section = header
            continue
        head, colon, rest = line.partition(":")
        inline_header = _section_header(head) if colon and rest.strip() else None
        if inline_header:
            section = inline_header
        for sentence in _SENTENCE_SPLIT.split(line):
            sentence = sentence.strip()
            if sentence:
                sentences.append((sentence, section))
    return sentences


def assess_sentences(skill: str, sentences: Iterable[tuple[str, str | None]]) -> list[SkillAssessment]:
    """
    Runs the rules on every sentence that names the skill.
    """
    skill = normalize_skill_name(skill)
    return [
        classify_rules(skill, sentence, section)
        for sentence, section in sentences
        if mentions_skill(sentence, skill)
    ]


def aggregate(skill: str, evidence: list[SkillAssessment]) -> SkillStatusResult:
    """
    Combines per-sentence assessments into one status for the skill.

    Positive evidence from Experience/Projects wins over Summary or Skills-list claims;
    otherwise an explicit limiting statement (WEAK/LEARNING/NONE) beats a bare mention.
    """
    skill = normalize_skill_name(skill)
    if not evidence:
        return SkillStatusResult(skill, HAS, SOURCE_DEFAULT, False, None, None, [])

    def pick(items: list[SkillAssessment]) -> SkillAssessment:
        return sorted(items, key=lambda item: (not item.matched, item.source == SOURCE_DEFAULT))[0]

    in_sections = [item for item in evidence if item.section in EVIDENCE_SECTIONS and item.status == HAS]
    limiting = [item for item in evidence if item.status != HAS and item.source != SOURCE_DEFAULT]
    if in_sections:
        chosen = pick(in_sections)
    elif limiting:
        best = max(_POSITIVITY[item.status] for item in limiting)
        chosen = pick([item for item in limiting if _POSITIVITY[item.status] == best])
    else:
        chosen = pick(evidence)
    return SkillStatusResult(
        skill=skill,
        status=chosen.status,
        source=chosen.source,
        matched=chosen.matched,
        evidence_sentence=chosen.evidence_sentence,
        section=chosen.section,
        evidence=evidence,
    )


def assess_cv_text(text: str, skills: Iterable[str]) -> dict[str, SkillStatusResult]:
    """
    Rule-based status for each skill, from the CV sentences that mention it.
    """
    sentences = split_cv_sentences(text)
    results: dict[str, SkillStatusResult] = {}
    for skill in skills:
        normalized = normalize_skill_name(skill)
        if normalized and normalized not in results:
            results[normalized] = aggregate(normalized, assess_sentences(normalized, sentences))
    return results


def unresolved_items(results: dict[str, SkillStatusResult]) -> list[SkillAssessment]:
    """
    Lists (skill, sentence) items no rule resolved and the LLM has not checked yet.
    """
    return [
        item
        for result in results.values()
        for item in result.evidence
        if item.source == SOURCE_DEFAULT and not item.llm_checked and item.evidence_sentence
    ]


# ---------------------------------------------------------------------------
# Storage format (CandidateProfile / Candidate fields)
# ---------------------------------------------------------------------------

def to_profile_fields(results: dict[str, SkillStatusResult]) -> dict[str, Any]:
    """
    Converts statuses to skills_detailed / learning_skills / negative_skills values.
    """
    details: list[dict[str, Any]] = []
    learning: list[str] = []
    negative: list[str] = []
    for skill, result in results.items():
        details.append({
            "name": skill,
            "status": STORED_STATUS[result.status],
            "confidence": STATUS_CONFIDENCE_CAP[result.status] if result.status != NONE else 0.0,
            "context": result.evidence_sentence,
            "source": result.source,
            "section": result.section,
            "evidence": [item.to_dict() for item in result.evidence][:10],
        })
        if result.status == LEARNING:
            learning.append(skill)
        elif result.status == NONE:
            negative.append(skill)
    return {"skills_detailed": details, "learning_skills": learning, "negative_skills": negative}


def results_from_details(details: Iterable[Any]) -> dict[str, SkillStatusResult]:
    """
    Rebuilds statuses from stored skills_detailed entries written by this module.
    """
    results: dict[str, SkillStatusResult] = {}
    for detail in details or []:
        if not isinstance(detail, dict) or "evidence" not in detail:
            continue
        skill = normalize_skill_name(str(detail.get("name") or ""))
        if not skill:
            continue
        evidence = [SkillAssessment.from_dict(item) for item in detail.get("evidence") or [] if isinstance(item, dict)]
        results[skill] = aggregate(skill, evidence) if evidence else SkillStatusResult(
            skill, stored_status_label(detail.get("status")) or HAS, str(detail.get("source") or SOURCE_DEFAULT),
            False, detail.get("context"), detail.get("section"), [],
        )
    return results


def stored_status_label(detail_status: str | None) -> str | None:
    """
    Maps a stored SkillDetail.status value back to a label.
    """
    value = getattr(detail_status, "value", detail_status)  # SkillStatus enum or plain str
    return LABEL_FROM_STORED.get(str(value or "").strip().lower())


def candidate_skill_statuses(candidate: Any) -> dict[str, SkillStatusResult]:
    """
    Returns the stored status per skill for a candidate, as the matcher should see it.

    Rows written by this module are read as stored. Legacy rows (no stored evidence)
    get the same rule assessment an upload would have stored, plus any legacy
    negative/learning lists.
    """
    stored = results_from_details(getattr(candidate, "skills_detailed", None) or [])
    if stored:
        return stored
    skills = list(getattr(candidate, "skills", None) or [])
    for detail in getattr(candidate, "skills_detailed", None) or []:
        if isinstance(detail, dict) and detail.get("name"):
            skills.append(str(detail["name"]))
    results = assess_cv_text(getattr(candidate, "raw_text", "") or "", skills)
    for detail in getattr(candidate, "skills_detailed", None) or []:
        if not isinstance(detail, dict):
            continue
        label = stored_status_label(detail.get("status"))
        skill = normalize_skill_name(str(detail.get("name") or ""))
        if skill and label in (NONE, LEARNING, WEAK):
            results[skill] = SkillStatusResult(skill, label, "legacy", True, detail.get("context"), None, [])
    for skill in getattr(candidate, "learning_skills", None) or []:
        normalized = normalize_skill_name(skill)
        if normalized and (normalized not in results or results[normalized].status == HAS):
            results[normalized] = SkillStatusResult(normalized, LEARNING, "legacy", True, None, None, [])
    for skill in getattr(candidate, "negative_skills", None) or []:
        normalized = normalize_skill_name(skill)
        if normalized:
            results[normalized] = SkillStatusResult(normalized, NONE, "legacy", True, None, None, [])
    return results


def raw_text_hash(text: str | None) -> str:
    """
    Hashes CV text so a refinement task can tell whether the CV changed meanwhile.
    """
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# LLM layer
# ---------------------------------------------------------------------------

LLM_PROMPT = """Classify how well a job candidate knows ONE named skill, using only the given CV sentence.
Labels:
HAS = has used the skill for real work. Example: skill ruby | "Wrote production services in Ruby."
WEAK = only basic or limited knowledge. Example: skill ruby | "Basic understanding of Ruby."
LEARNING = is studying it or plans to learn it. Example: skill ruby | "Currently taking a Ruby course."
NONE = says they do not have it. Example: skill ruby | "I have never used Ruby."
Judge only the named skill; other skills in the same sentence do not matter.
Return only a JSON object mapping each item number to one label, like {{"1": "HAS", "2": "NONE"}}.

Items:
{items}
"""


class LLMUnavailable(RuntimeError):
    """The local LLM could not be reached or answered with unusable output."""


def build_llm_prompt(items: list[tuple[str, str]]) -> str:
    """
    Builds the batched prompt for (skill, sentence) items.
    """
    lines = [
        f'{index}. skill: {skill} | sentence: "{" ".join(sentence.split())[:400]}"'
        for index, (skill, sentence) in enumerate(items, start=1)
    ]
    return LLM_PROMPT.format(items="\n".join(lines))


def parse_llm_labels(raw: str | dict[str, Any], count: int) -> dict[int, str]:
    """
    Strictly parses the model output: a JSON object of item number -> known label.
    Anything else is ignored, so the caller keeps its rule result for that item.
    """
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    labels: dict[int, str] = {}
    for key, value in data.items():
        try:
            index = int(str(key).strip().rstrip("."))
        except ValueError:
            continue
        label = str(value).strip().upper() if isinstance(value, str) else ""
        if 1 <= index <= count and label in LABELS:
            labels[index] = label
    return labels


def accept_llm_label(label: str, sentence: str) -> bool:
    """
    Rejects an LLM NONE for a sentence with no negation at all ("JavaScript (advanced)").
    """
    return label != NONE or bool(_NEGATION_CUE.search(normalize_for_rules(sentence)))


async def llm_classify(items: list[tuple[str, str]], timeout: float | None = None) -> tuple[dict[int, str], float]:
    """
    Sends (skill, sentence) items to the local model in batches.

    Returns ({item number: label}, elapsed seconds). Raises LLMUnavailable when
    Ollama is down, times out, or returns no usable JSON for any batch.
    """
    import httpx

    if not items:
        return {}, 0.0
    timeout = timeout or settings.skill_status_llm_timeout_seconds
    batch_size = max(1, settings.skill_status_llm_batch_size)
    labels: dict[int, str] = {}
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=httpx.Timeout(timeout, connect=3.0)) as client:
            for offset in range(0, len(items), batch_size):
                batch = items[offset:offset + batch_size]
                response = await client.post("/api/generate", json={
                    "model": settings.ollama_parsing_model,
                    "prompt": build_llm_prompt(batch),
                    "format": "json",
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {"temperature": 0},
                })
                response.raise_for_status()
                batch_labels = parse_llm_labels(response.json().get("response", ""), len(batch))
                labels.update({offset + index: label for index, label in batch_labels.items()})
    except Exception as exc:
        raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc
    if not labels:
        raise LLMUnavailable("LLM returned no valid labels")
    return labels, time.monotonic() - started


def apply_llm_labels(items: list[SkillAssessment], labels: dict[int, str]) -> int:
    """
    Writes valid LLM labels onto unresolved items (1-based numbering). Returns changes.
    """
    changed = 0
    for index, item in enumerate(items, start=1):
        item.llm_checked = True
        label = labels.get(index)
        if label is None or not accept_llm_label(label, item.evidence_sentence or ""):
            continue
        item.status = label
        item.source = SOURCE_LLM
        item.matched = True
        changed += 1
    return changed


async def refine_results(results: dict[str, SkillStatusResult]) -> dict[str, Any]:
    """
    Runs the LLM on unresolved items and re-aggregates the affected skills in place.
    """
    pending = unresolved_items(results)[: max(0, settings.skill_status_llm_max_items)]
    if not pending:
        return {"status": "noop", "items": 0}
    labels, elapsed = await llm_classify([(item.skill, item.evidence_sentence or "") for item in pending])
    changed = apply_llm_labels(pending, labels)
    for skill in {item.skill for item in pending}:
        results[skill] = aggregate(skill, results[skill].evidence)
    return {"status": "refined", "items": len(pending), "labels": len(labels), "changed": changed, "seconds": round(elapsed, 2)}
