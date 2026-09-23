"""
Benchmark RAG partial-skill credit: OLD whole-CV similarity vs NEW indicator logic.

Usage (from backend/):
    python scripts/benchmark_rag_enrichment.py                 # report with configured T
    python scripts/benchmark_rag_enrichment.py --sweep         # also sweep T on the tuning split
    python scripts/benchmark_rag_enrichment.py --hide-heldout  # never look at held-out while tuning

Each case is a CV snippet parsed with the rule-based parser. A case whose skill
is already matched by the exact / synonym / ESCO / text stages is not a RAG
case and is excluded from the metrics. Runs against a throwaway SQLite
database, never backend/app.db.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
CASES_PATH = BACKEND_DIR / "tests" / "fixtures" / "rag_skill_cases.jsonl"

# --pad wraps every snippet in generic CV lines (no indicators, no skill names)
# so a decision cannot depend on the snippet being the only text in the CV.
PAD_BEFORE = "\n".join([
    "Software Engineer",
    "Passionate about solving complex technical challenges and delivering high-quality software solutions.",
    "Experience",
    "",
])
PAD_AFTER = "\n".join([
    "",
    "Documented deployment steps for the team.",
    "Collaborated with product managers to plan features and review designs.",
    "Education",
    "Bachelor of Science in Computer Science, 2021.",
    "Relevant coursework: Operating Systems, Algorithms, Discrete Mathematics, Software Engineering.",
    "Languages: English, French, Arabic.",
])


def _configure_environment(provider: str) -> Path:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ["EMBEDDING_PROVIDER"] = provider
    os.environ.setdefault("LLM_PROVIDER", "rule")
    os.environ["RAG_AUTO_SEED"] = "false"
    db_path = Path(tempfile.mkdtemp(prefix="rag_bench_")) / "bench.sqlite"
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path.as_posix()}"
    for path in (BACKEND_DIR, BACKEND_DIR / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return db_path


@dataclass
class CaseResult:
    case: dict
    excluded: str | None = None
    old_initial: bool = False
    old_seeded: bool = False
    old_similarity: float | None = None
    indicators: list[str] = field(default_factory=list)
    similarity: float | None = None
    best_chunk: str | None = None
    negated: bool = False
    new: bool = False

    def new_decision(self, threshold: float, min_indicators: int) -> bool:
        if self.negated or not self.indicators:
            return False
        if len(self.indicators) >= min_indicators:
            return True
        return self.similarity is not None and self.similarity >= threshold


def _metrics(pairs: list[tuple[bool, bool]]) -> dict:
    tp = sum(1 for pred, gold in pairs if pred and gold)
    fp = sum(1 for pred, gold in pairs if pred and not gold)
    fn = sum(1 for pred, gold in pairs if not pred and gold)
    tn = sum(1 for pred, gold in pairs if not pred and not gold)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else 0.0
    f05 = (1.25 * precision * recall / (0.25 * precision + recall)) if precision and recall else 0.0
    return {"n": len(pairs), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1, "f0_5": f05}


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:5.3f}"


def _snippet_text(case: dict, pad: bool) -> str:
    return f"{PAD_BEFORE}{case['cv_snippet']}{PAD_AFTER}" if pad else case["cv_snippet"]


async def _evaluate(cases: list[dict], pad: bool = False) -> list[CaseResult]:
    from app.core.config import settings
    from app.core.db import SessionLocal, init_db
    from app.models.candidate import Candidate
    from app.models.job import Job
    from app.services.cv_parser import parse_cv_text
    from app.services.hybrid_matcher import HybridMatchingEngine, SkillMatchResult
    from app.services.rag import get_skill_definitions, sync_knowledge_base
    from app.services.skill_catalog import normalize_skill_name
    from _legacy_rag import LEGACY_THRESHOLD, legacy_initial_definitions, legacy_similarity

    await init_db()
    async with SessionLocal() as session:
        report = await sync_knowledge_base(session)
    print(f"Seeded temp knowledge base: {report.total} docs with {report.embedding_model} in {report.elapsed_ms:.0f} ms")

    # Collect raw features once with semantic credit disabled; decisions for any
    # threshold are then computed offline (and re-checked through the engine).
    settings.rag_chunk_similarity_threshold = 2.0
    engine = HybridMatchingEngine()
    results: list[CaseResult] = []
    async with SessionLocal() as session:
        for case in cases:
            skill = normalize_skill_name(case["required_skill"])
            profile = parse_cv_text(_snippet_text(case, pad))
            candidate = Candidate(
                id=f"bench-{case['id']}-{uuid.uuid4().hex[:6]}",
                full_name="Benchmark Candidate",
                email=None,
                phone=None,
                skills=profile.skills,
                skills_detailed=[s.model_dump() if hasattr(s, "model_dump") else s for s in (profile.skills_detailed or [])],
                experience=profile.experience,
                education=profile.education,
                projects=profile.projects,
                raw_text=profile.raw_text,
                negative_skills=profile.negative_skills or None,
                learning_skills=profile.learning_skills or None,
                uncatalogued_skills=profile.uncatalogued_skills or None,
            )
            job = Job(id=f"bench-job-{case['id']}", title="Benchmark", description="Benchmark job",
                      required_skills=[skill], optional_skills=[], seniority="mid")
            result = CaseResult(case=case)

            upstream = await engine._compute_skill_match(job, candidate, rag_session=None)
            if upstream.matched_required:
                result.excluded = f"matched upstream ({upstream.matched_required[0].match_type})"
                results.append(result)
                continue

            initial_definition = legacy_initial_definitions([skill]).get(skill)
            if initial_definition:
                result.old_initial = await legacy_similarity(engine, initial_definition, candidate) > LEGACY_THRESHOLD
            seeded_definition = (await get_skill_definitions([skill], session=session)).get(skill)
            if seeded_definition:
                result.old_similarity = await legacy_similarity(engine, seeded_definition, candidate)
                result.old_seeded = result.old_similarity > LEGACY_THRESHOLD

            skill_result = SkillMatchResult(missing_required=[skill])
            await engine._apply_rag_enrichment(skill_result, candidate, session)
            info = skill_result.rag_evidence.get(skill) or skill_result.rag_rejected.get(skill) or {}
            result.indicators = list(info.get("indicators") or [])
            result.similarity = info.get("similarity")
            result.best_chunk = info.get("best_chunk")
            result.negated = "negated" in str(info.get("reason", ""))
            if skill in skill_result.rag_evidence and result.similarity is None:
                # >=2 indicators without semantic evidence (hash provider).
                result.similarity = None
            results.append(result)
    return results


async def _verify_through_engine(results: list[CaseResult], threshold: float, pad: bool = False) -> int:
    """
    Re-runs NEW with the chosen threshold through the real engine; returns mismatches.
    """
    from app.core.config import settings
    from app.core.db import SessionLocal
    from app.models.candidate import Candidate
    from app.services.cv_parser import parse_cv_text
    from app.services.hybrid_matcher import HybridMatchingEngine, SkillMatchResult
    from app.services.skill_catalog import normalize_skill_name

    settings.rag_chunk_similarity_threshold = threshold
    engine = HybridMatchingEngine()
    mismatches = 0
    async with SessionLocal() as session:
        for result in results:
            if result.excluded:
                continue
            skill = normalize_skill_name(result.case["required_skill"])
            profile = parse_cv_text(_snippet_text(result.case, pad))
            candidate = Candidate(
                id=f"verify-{result.case['id']}", full_name="Benchmark Candidate", email=None, phone=None,
                skills=profile.skills, experience=profile.experience, education=profile.education,
                projects=profile.projects, raw_text=profile.raw_text,
                negative_skills=profile.negative_skills or None, learning_skills=profile.learning_skills or None,
            )
            skill_result = SkillMatchResult(missing_required=[skill])
            await engine._apply_rag_enrichment(skill_result, candidate, session)
            result.new = skill in skill_result.rag_enriched_skills
            if result.new != result.new_decision(threshold, settings.rag_min_indicators):
                mismatches += 1
    return mismatches


def _sweep(results: list[CaseResult], min_indicators: int) -> tuple[float, list[tuple[float, dict]]]:
    tuning = [r for r in results if not r.excluded and r.case["split"] == "tuning"]
    rows: list[tuple[float, dict]] = []
    for step in range(20, 81):
        threshold = step / 100
        rows.append((threshold, _metrics([(r.new_decision(threshold, min_indicators), r.case["should_credit"]) for r in tuning])))
    # FP-averse selection: best F0.5 on tuning; ties go to the higher (stricter) threshold.
    best = max(rows, key=lambda row: (round(row[1]["f0_5"], 6), row[0]))
    return best[0], rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="sentence-transformers")
    parser.add_argument("--threshold", type=float, default=None, help="Override the configured T")
    parser.add_argument("--sweep", action="store_true", help="Sweep T on the tuning split and use the selected value")
    parser.add_argument("--hide-heldout", action="store_true", help="Do not report held-out results")
    parser.add_argument("--pad", action="store_true", help="Wrap snippets in generic CV lines (robustness check)")
    parser.add_argument("--json", type=Path, default=None, help="Write per-case results as JSON")
    args = parser.parse_args()

    _configure_environment(args.provider)
    from app.core.config import settings

    cases = [json.loads(line) for line in CASES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    started = time.perf_counter()
    results = asyncio.run(_evaluate(cases, pad=args.pad))
    min_indicators = settings.rag_min_indicators

    configured = type(settings)().rag_chunk_similarity_threshold
    threshold = args.threshold if args.threshold is not None else configured
    if args.sweep:
        selected, rows = _sweep(results, min_indicators)
        print("\nThreshold sweep on TUNING split (single-indicator cases only depend on T):")
        print("   T    prec   rec    F1     F0.5   FP  FN")
        previous = None
        for value, m in rows:
            signature = (m["tp"], m["fp"])
            if signature != previous or value == selected:
                marker = "  <- selected" if value == selected else ""
                print(f"  {value:.2f}  {_fmt(m['precision'])}  {_fmt(m['recall'])}  {m['f1']:.3f}  {m['f0_5']:.3f}  {m['fp']:>2}  {m['fn']:>2}{marker}")
            previous = signature
        if args.threshold is None:
            threshold = selected
        print(f"Selected T = {selected:.2f} (max F0.5 on tuning, ties -> stricter). Using T = {threshold:.2f}.")

    mismatches = asyncio.run(_verify_through_engine(results, threshold, pad=args.pad))
    elapsed = time.perf_counter() - started

    excluded = [r for r in results if r.excluded]
    counted = [r for r in results if not r.excluded]
    splits = ["tuning", "all"] if args.hide_heldout else ["tuning", "heldout", "all"]
    print(f"\nCases: {len(results)} total, {len(excluded)} excluded (not RAG cases), {len(counted)} counted. "
          f"Provider={settings.embedding_provider}, T={threshold:.2f}, min_indicators={min_indicators}, "
          f"padded={args.pad}")
    for r in excluded:
        print(f"  excluded {r.case['id']} [{r.case['kind']}] {r.case['required_skill']}: {r.excluded}")

    methods = {
        "OLD (INITIAL_KNOWLEDGE, whole-CV > 0.75)": lambda r: r.old_initial,
        "OLD (seeded KB, whole-CV > 0.75)": lambda r: r.old_seeded,
        "NEW (indicators + chunk similarity)": lambda r: r.new,
    }
    print("\n| method | split | n | TP | FP | FN | TN | precision | recall | F1 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, predict in methods.items():
        for split in splits:
            rows = [r for r in counted if split == "all" or r.case["split"] == split]
            if args.hide_heldout and split == "all":
                rows = [r for r in rows if r.case["split"] != "heldout"]
            m = _metrics([(predict(r), r.case["should_credit"]) for r in rows])
            print(f"| {name} | {split} | {m['n']} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | "
                  f"{_fmt(m['precision'])} | {_fmt(m['recall'])} | {m['f1']:.3f} |")

    visible = [r for r in counted if not (args.hide_heldout and r.case["split"] == "heldout")]
    for name, predict in methods.items():
        false_positives = [r for r in visible if predict(r) and not r.case["should_credit"]]
        print(f"\nFalse positives - {name}: {len(false_positives)}")
        for r in false_positives:
            print(f"  {r.case['id']} [{r.case['split']}/{r.case['kind']}] {r.case['required_skill']}: "
                  f"indicators={r.indicators} sim={r.similarity} old_sim={_fmt(r.old_similarity)} :: {r.case['cv_snippet']}")
    false_negatives = [r for r in visible if not r.new and r.case["should_credit"]]
    print(f"\nFalse negatives - NEW: {len(false_negatives)}")
    for r in false_negatives:
        sim = "n/a" if r.similarity is None else f"{r.similarity:.3f}"
        print(f"  {r.case['id']} [{r.case['split']}] {r.case['required_skill']}: indicators={r.indicators} sim={sim}")

    print(f"\nEngine re-check mismatches: {mismatches}. Elapsed {elapsed:.1f}s.")
    if args.json:
        args.json.write_text(json.dumps([{
            **r.case, "excluded": r.excluded, "old_initial": r.old_initial, "old_seeded": r.old_seeded,
            "old_similarity": r.old_similarity, "indicators": r.indicators, "similarity": r.similarity,
            "best_chunk": r.best_chunk, "negated": r.negated, "new": r.new,
        } for r in results], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
