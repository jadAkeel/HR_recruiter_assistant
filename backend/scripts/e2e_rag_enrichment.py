"""
End-to-end check of RAG partial credit on a temporary copy of a SQLite database.

Steps: copy the DB (read-only source), sync the knowledge base, parse and
insert the CVs from cvs_to_upload, create jobs with umbrella skills, then match
every job with the OLD (legacy whole-CV) and NEW engines and compare scores.

Usage (from backend/):
    python scripts/e2e_rag_enrichment.py --source-db ../../backend/app.db
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
import tempfile
import time
import uuid
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]

JOBS = [
    {
        "title": "NLP Engineer",
        "description": "Build language understanding features: classify support tickets, extract entities and summarize documents with transformer models.",
        "required_skills": ["nlp", "python"],
        "optional_skills": ["deep learning", "docker"],
        "seniority": "junior",
    },
    {
        "title": "Machine Learning Engineer",
        "description": "Train, evaluate and ship predictive models for churn and fraud using Python and SQL.",
        "required_skills": ["machine learning", "python", "sql"],
        "optional_skills": ["data engineering", "cloud computing"],
        "seniority": "junior",
    },
    {
        "title": "DevOps Engineer",
        "description": "Own CI/CD pipelines, container platforms and infrastructure as code for our cloud services.",
        "required_skills": ["devops", "kubernetes", "ci/cd"],
        "optional_skills": ["aws", "linux"],
        "seniority": "junior",
    },
]


def _copy_database(source: Path, destination: Path) -> None:
    """
    Copies a SQLite database with the backup API; the source is opened read-only.
    """
    src = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    dst = sqlite3.connect(destination)
    with dst:
        src.backup(dst)
    src.close()
    dst.close()


def _configure(db_path: Path, provider: str) -> None:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ["EMBEDDING_PROVIDER"] = provider
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path.as_posix()}"
    os.environ["RAG_AUTO_SEED"] = "false"
    os.environ.setdefault("LLM_PROVIDER", "rule")
    for path in (BACKEND_DIR, BACKEND_DIR / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


async def _run(cv_dir: Path, limit: int | None) -> dict:
    from app.core.db import SessionLocal, init_db
    from app.models.candidate import Candidate
    from app.models.job import Job
    from app.services.cv_parser import extract_text, parse_cv_text
    from app.services.hybrid_matcher import HybridMatchingEngine
    from app.services.rag import sync_knowledge_base
    from app.services.vector_store import VectorStore
    from _legacy_rag import LegacyRagEngine

    report: dict = {"timings": {}}
    await init_db()
    async with SessionLocal() as session:
        first = await sync_knowledge_base(session)
        second = await sync_knowledge_base(session)
    report["seed"] = {"first": first.to_dict(), "second": second.to_dict()}

    files = sorted(p for p in cv_dir.iterdir() if p.suffix.lower() in {".pdf", ".docx", ".txt"})[:limit]
    started = time.perf_counter()
    candidates: list[Candidate] = []
    names: dict[str, str] = {}
    async with SessionLocal() as session:
        for path in files:
            text = extract_text(path.name, path.read_bytes())
            if not text:
                continue
            profile = parse_cv_text(text)
            candidate = Candidate(
                id=str(uuid.uuid4()),
                full_name=profile.full_name,
                email=profile.email,
                phone=profile.phone,
                skills=profile.skills,
                skills_detailed=[s.model_dump() if hasattr(s, "model_dump") else s for s in (profile.skills_detailed or [])],
                experience=profile.experience,
                education=profile.education,
                projects=profile.projects,
                raw_text=profile.raw_text,
                total_years_experience=profile.total_years_experience,
                negative_skills=profile.negative_skills or None,
                learning_skills=profile.learning_skills or None,
                uncatalogued_skills=profile.uncatalogued_skills or None,
                cv_file_name=path.name,
            )
            session.add(candidate)
            candidates.append(candidate)
            names[candidate.id] = path.stem
        await session.commit()
    report["timings"]["parse_and_insert_s"] = round(time.perf_counter() - started, 2)
    report["candidates"] = len(candidates)

    jobs_report = []
    async with SessionLocal() as session:
        for spec in JOBS:
            job = Job(id=str(uuid.uuid4()), **spec)
            session.add(job)
            await session.commit()
            runs = {}
            # Warm run so model loading and candidate embedding caching are not
            # attributed to whichever engine happens to go first.
            await HybridMatchingEngine().match(job, candidates, top_k=len(candidates), enable_cross_encoder=False,
                                              vector_store=VectorStore(session), rag_session=session)
            for label, engine_cls in (("old", LegacyRagEngine), ("new", HybridMatchingEngine)):
                timings = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    results = await engine_cls().match(job, candidates, top_k=len(candidates), enable_cross_encoder=False,
                                                       vector_store=VectorStore(session), rag_session=session)
                    timings.append(time.perf_counter() - t0)
                runs[label] = ({r.candidate_id: r for r in results}, sorted(timings)[1])
            old, old_time = runs["old"]
            new, new_time = runs["new"]
            rows = []
            for cid, new_result in new.items():
                old_result = old[cid]
                delta = new_result.final_score - old_result.final_score
                if not new_result.skill_match.rag_evidence and not old_result.skill_match.rag_enriched_skills and abs(delta) < 1e-9:
                    continue
                rows.append({
                    "candidate": names[cid],
                    "old_score": round(old_result.final_score * 100, 1),
                    "new_score": round(new_result.final_score * 100, 1),
                    "delta_points": round(delta * 100, 1),
                    "flag_over_5_points": abs(delta) > 0.05,
                    "old_rag_skills": old_result.skill_match.rag_enriched_skills,
                    "new_rag_skills": new_result.skill_match.rag_enriched_skills,
                    "evidence": new_result.skill_match.rag_evidence,
                })
            rows.sort(key=lambda row: -abs(row["delta_points"]))
            rejected = {}
            for new_result in new.values():
                for skill, info in new_result.skill_match.rag_rejected.items():
                    rejected.setdefault(skill, 0)
                    rejected[skill] += 1
            jobs_report.append({
                "job": spec["title"],
                "required": spec["required_skills"],
                "old_rag_credited": sum(1 for r in old.values() if r.skill_match.rag_enriched_skills),
                "new_rag_credited": sum(1 for r in new.values() if r.skill_match.rag_enriched_skills),
                "near_misses_by_skill": rejected,
                "match_seconds_old": round(old_time, 3),
                "match_seconds_new": round(new_time, 3),
                "changes": rows,
            })
    report["jobs"] = jobs_report
    return report


def _print(report: dict) -> None:
    seed = report["seed"]
    print(f"Knowledge sync #1: {seed['first']}")
    print(f"Knowledge sync #2: {seed['second']}")
    print(f"Candidates uploaded: {report['candidates']} ({report['timings']['parse_and_insert_s']}s)")
    for job in report["jobs"]:
        print(f"\n=== {job['job']} (required: {', '.join(job['required'])}) ===")
        print(f"RAG-credited candidates: old={job['old_rag_credited']} new={job['new_rag_credited']}; "
              f"match time old={job['match_seconds_old']}s new={job['match_seconds_new']}s; "
              f"refused near-misses={job['near_misses_by_skill']}")
        for row in job["changes"]:
            flag = "  <-- REVIEW (>5 pts)" if row["flag_over_5_points"] else ""
            print(f"- {row['candidate']}: {row['old_score']} -> {row['new_score']} ({row['delta_points']:+.1f}){flag}")
            for skill, ev in row["evidence"].items():
                sim = "n/a" if ev["similarity"] is None else f"{ev['similarity']:.3f}"
                print(f"    {skill}: indicators={ev['indicators']} sim={sim} rule='{ev['rule']}'")
                print(f"      best chunk: {ev['best_chunk']}")
            if row["old_rag_skills"]:
                print(f"    old RAG credit: {row['old_rag_skills']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-db", type=Path, required=True, help="SQLite DB to copy (never modified)")
    parser.add_argument("--cv-dir", type=Path, default=BACKEND_DIR / "cvs_to_upload")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--provider", default="sentence-transformers")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="rag_e2e_"))
    db_copy = workdir / "app_copy.db"
    _copy_database(args.source_db.resolve(), db_copy)
    print(f"Working on temp copy: {db_copy}")
    _configure(db_copy, args.provider)
    report = asyncio.run(_run(args.cv_dir, args.limit))
    _print(report)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
