import asyncio
import uuid
from pathlib import Path

import pytest

from app.core.config import settings
from app.core.db import SessionLocal, init_db
from app.models.candidate import Candidate


def _assert_off_event_loop() -> None:
    """
    Fails when called on a thread that is running an asyncio event loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise AssertionError("CPU-bound CV work ran on the event loop thread")


def test_pending_cv_uploads_live_inside_cv_storage() -> None:
    """
    Checks that queued uploads are saved under CV_STORAGE_PATH, the directory a
    separate worker container shares with the API (docker-compose.prod.yml mounts
    the cv_uploads volume there). A sibling directory would be container-local.
    """
    from app.api import candidates as candidates_api

    assert candidates_api.PENDING_CV_STORAGE.is_relative_to(candidates_api.CV_STORAGE)


@pytest.mark.asyncio
async def test_worker_processes_a_pending_cv_off_the_event_loop(monkeypatch, tmp_path) -> None:
    """
    Checks that the queue worker reads a pending upload, parses it in a worker
    thread, stores the candidate, and removes the pending file.
    """
    import app.main as main_module
    from app.services import cv_parser, task_queue

    captured: dict = {}

    async def _capture_worker(process_func, refine_func=None):
        captured["process"] = process_func

    monkeypatch.setattr(task_queue, "run_cv_worker", _capture_worker)
    monkeypatch.setattr(settings, "cv_storage_path", str(tmp_path / "cvs"))

    real_extract, real_parse = cv_parser.extract_text, cv_parser.parse_cv_text

    def _extract(file_name, content):
        _assert_off_event_loop()
        return real_extract(file_name, content)

    def _parse(text):
        _assert_off_event_loop()
        return real_parse(text)

    monkeypatch.setattr(cv_parser, "extract_text", _extract)
    monkeypatch.setattr(cv_parser, "parse_cv_text", _parse)

    await init_db()
    await main_module._cv_worker()

    email = f"worker-{uuid.uuid4().hex[:8]}@example.com"
    pending = tmp_path / "cvs" / "pending" / f"{uuid.uuid4()}.txt"
    pending.parent.mkdir(parents=True)
    pending.write_bytes(f"Worker Candidate\n{email}\nSkills\nPython, FastAPI".encode())

    result = await captured["process"](
        cv_text=None,
        file_name="worker.txt",
        use_llm=False,
        file_path=str(pending),
    )

    assert result["status"] == "created"
    assert not pending.exists()
    assert Path(settings.cv_storage_path, f"{result['candidate_id']}.txt").exists()
    async with SessionLocal() as session:
        candidate = await session.get(Candidate, result["candidate_id"])
    assert candidate is not None and candidate.email == email
    assert "python" in candidate.skills


@pytest.mark.asyncio
async def test_rule_based_enhanced_parse_runs_off_the_event_loop(monkeypatch) -> None:
    """
    Checks that parse_async keeps sentence parsing and rule matching in threads.
    """
    from app.services.enhanced_cv_parser import get_simple_cv_parser

    parser = get_simple_cv_parser()
    real_nlp, real_rules = parser._parse_text_for_nlp, parser._extract_skills_rule_based

    def _nlp(text):
        _assert_off_event_loop()
        return real_nlp(text)

    def _rules(*args):
        _assert_off_event_loop()
        return real_rules(*args)

    monkeypatch.setattr(parser, "_parse_text_for_nlp", _nlp)
    monkeypatch.setattr(parser, "_extract_skills_rule_based", _rules)

    profile = await parser.parse_async("Thread Candidate\nthread@example.com\nSkills\nPython, Docker")

    assert "python" in profile.skills
