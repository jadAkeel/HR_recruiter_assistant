from __future__ import annotations

import logging
import re

from app.schemas.job import JobProfile
from app.services.skill_catalog import SKILL_KEYWORDS, extract_catalog_skills, normalize_text_for_skill_matching, skill_in_text

logger = logging.getLogger(__name__)

_REQUIRED_KEYWORDS = (
    r"requirements|required skills|required qualifications|required|must[- ]have|minimum qualifications"
)
_OPTIONAL_KEYWORDS = (
    r"nice[- ]to[- ]have|good to have|preferred qualifications|preferred|bonus|desirable"
)
_OTHER_SECTION_KEYWORDS = r"responsibilities|about|benefits|overview|summary"

REQUIREMENT_HEADERS = re.compile(rf"^({_REQUIRED_KEYWORDS})\b", re.IGNORECASE)
OPTIONAL_HEADERS = re.compile(rf"^({_OPTIONAL_KEYWORDS})\b", re.IGNORECASE)
SECTION_BOUNDARY_HEADERS = re.compile(
    rf"^({_REQUIRED_KEYWORDS}|{_OPTIONAL_KEYWORDS}|{_OTHER_SECTION_KEYWORDS})\b",
    re.IGNORECASE,
)
# "Keyword:" mid-paragraph starts a new section, e.g. "Backend Developer Requirements: Python".
INLINE_HEADER_SPLIT = re.compile(
    rf"(?=\b(?:{_REQUIRED_KEYWORDS}|{_OPTIONAL_KEYWORDS}|{_OTHER_SECTION_KEYWORDS})\s*:)",
    re.IGNORECASE,
)
MAX_TITLE_LENGTH = 120
SENIORITY_PATTERNS = {
    "junior": re.compile(r"\b(junior|entry level|associate)\b", re.IGNORECASE),
    "mid": re.compile(r"\b(mid|intermediate)\b", re.IGNORECASE),
    "senior": re.compile(r"\b(senior|lead|principal|staff)\b", re.IGNORECASE),
}


BONUS_PHRASE_PATTERN = re.compile(
    r"\b(is a plus|nice[- ]to[- ]have|preferred|bonus|plus|good to have|desirable|advantage)",
    re.IGNORECASE,
)
# Sentence boundary: punctuation followed by whitespace, so "node.js" or "3.5" stay intact.
CLAUSE_SPLIT_PATTERN = re.compile(r"(?<=[.;!?])\s+")


def parse_job_description(text: str) -> JobProfile:
    """
    Parses a job description into title, required skills, optional skills, and
    seniority.
    """
    normalized = _normalize_text(text)
    required = _extract_section_skills(text, REQUIREMENT_HEADERS)
    optional = _extract_section_skills(text, OPTIONAL_HEADERS)

    all_skills = _extract_skills(normalized)

    if BONUS_PHRASE_PATTERN.search(normalized):
        optional = sorted(set(optional) | _extract_bonus_skills(text))

    if not required:
        required = [s for s in all_skills if s not in set(optional)]

    if not required and not optional:
        required = all_skills

    if optional:
        optional = sorted(skill for skill in set(optional) if skill not in set(required))
    required = sorted(set(required))

    seniority = _detect_seniority(text)

    profile = JobProfile(
        title=_extract_title(text),
        description=text,
        required_skills=required,
        optional_skills=optional,
        seniority=seniority,
    )
    logger.info("Job parsed", extra={"required": len(required), "optional": len(optional)})
    return profile


def _extract_bonus_skills(text: str) -> set[str]:
    """
    Extracts skills from sentences marked as bonus or preferred, so a single-paragraph
    description does not turn its required skills into optional ones.
    """
    bonus_skills: set[str] = set()
    for segment in _segments(text):
        if segment and BONUS_PHRASE_PATTERN.search(segment):
            segment_normalized = _normalize_text(segment)
            for skill in SKILL_KEYWORDS:
                if skill_in_text(skill, segment_normalized):
                    bonus_skills.add(skill)
    return bonus_skills


def _segments(text: str) -> list[str]:
    """
    Splits a job description into sentence-level segments, starting a new segment at
    every sentence boundary and inline "Header:" keyword. Blank lines become "" so
    they still end a section.
    """
    segments: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            segments.append("")
            continue
        for clause in CLAUSE_SPLIT_PATTERN.split(line):
            for piece in INLINE_HEADER_SPLIT.split(clause):
                piece = piece.strip()
                if piece:
                    segments.append(piece)
    return segments


def _extract_title(text: str) -> str | None:
    """
    Uses the first sentence of the description as the likely job title, stopping at
    any section header so a one-paragraph description does not become the title.
    """
    for segment in _segments(text):
        if not segment:
            continue
        if SECTION_BOUNDARY_HEADERS.search(segment):
            return None
        title = segment.rstrip(" .;:!?-").strip()
        if len(title) > MAX_TITLE_LENGTH:
            title = title[:MAX_TITLE_LENGTH].rsplit(" ", 1)[0].rstrip(" ,;:-")
        return title or None
    return None


def _normalize_text(text: str) -> str:
    """
    Normalizes job text for skill matching.
    """
    return normalize_text_for_skill_matching(text)


def _extract_skills(normalized_text: str) -> list[str]:
    """
    Extracts catalog skills from normalized job text.
    """
    return extract_catalog_skills(normalized_text)


def _extract_section_skills(text: str, header_pattern: re.Pattern[str]) -> list[str]:
    """
    Extracts skills from a labeled job description section.
    """
    in_section = False
    collected: list[str] = []
    for line in _segments(text):
        if header_pattern.search(line):
            in_section = True
            remainder = header_pattern.sub("", line, count=1).strip(" :-")
            if remainder:
                collected.append(remainder)
            continue
        if in_section and SECTION_BOUNDARY_HEADERS.search(line):
            break
        if in_section and not line:
            break
        if in_section:
            collected.append(line)

    normalized = _normalize_text(" ".join(collected))
    return _extract_skills(normalized)


def _detect_seniority(text: str) -> str | None:
    """
    Detects job seniority from common seniority phrases.
    """
    for label, pattern in SENIORITY_PATTERNS.items():
        if pattern.search(text):
            return label
    return None
