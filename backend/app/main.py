from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.core.db import init_db
from app.core.logging import configure_logging, shutdown_logging
from app.core.redis import close_redis, get_redis
from app.core.security import RateLimitMiddleware, SecurityHeadersMiddleware

logger = logging.getLogger(__name__)


def _resolve_keep_alive_url() -> str | None:
    """
    Resolves the public URL used to keep Render free services warm.
    """
    if settings.keep_alive_url:
        return settings.keep_alive_url
    render_host = os.environ.get("RENDER_EXTERNAL_HOSTNAME")
    if render_host:
        return f"https://{render_host}{settings.api_prefix}/health"
    return None


def _should_run_keep_alive() -> bool:
    """
    Enables keep-alive when explicitly configured or when running on Render.
    """
    return settings.keep_alive_enabled or bool(os.environ.get("RENDER_EXTERNAL_HOSTNAME"))


async def _keep_alive_worker() -> None:
    """
    Periodically pings the public health endpoint to reduce Render idle spin-downs.
    """
    url = _resolve_keep_alive_url()
    if not url:
        logger.warning("Keep-alive enabled but no public URL is configured")
        return

    interval = max(settings.keep_alive_interval_seconds, 60.0)
    logger.info("Keep-alive scheduled every %.0f seconds via %s", interval, url)
    await asyncio.sleep(interval)
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        while True:
            try:
                response = await client.get(url)
                if response.is_success:
                    logger.debug("Keep-alive ping returned %s from %s", response.status_code, url)
                else:
                    logger.warning("Keep-alive ping returned %s from %s", response.status_code, url)
            except Exception:
                logger.warning("Keep-alive ping failed for %s", url, exc_info=True)
            await asyncio.sleep(interval)


async def _warm_embedding_model() -> None:
    """
    Loads the local embedding model at startup, off the event loop, so the first
    request is fast and a model that cannot load is reported in the startup log.
    """
    if settings.embedding_provider != "sentence-transformers":
        return
    from app.services.embedding import (
        embedding_runtime_status,
        get_multilingual_embedding_service,
        get_standard_embedding_service,
    )

    loader = get_multilingual_embedding_service if settings.use_multilingual_embedding else get_standard_embedding_service
    try:
        await asyncio.to_thread(loader)
    except Exception:
        logger.exception("Embedding model failed to load at startup")
        return
    for model, status in embedding_runtime_status().items():
        logger.info("Embedding model %s served by %s", model, status["active_provider"])


async def _ensure_initial_owner() -> None:
    """
    Ensures a configured owner account exists before serving requests.
    """
    from app.core.db import SessionLocal
    from app.services.auth import ensure_initial_owner

    async with SessionLocal() as session:
        owner = await ensure_initial_owner(session)
        if owner is not None:
            logger.info("Initial owner account is ready: %s", owner.email)


def _start_knowledge_sync() -> asyncio.Task | None:
    """
    Syncs the RAG knowledge base in the background so embedding never delays startup.
    """
    if not settings.rag_auto_seed:
        return None
    from app.services.rag import seed_knowledge_base_on_startup

    logger.info("RAG knowledge base sync started in background")
    return asyncio.create_task(seed_knowledge_base_on_startup())


# Background worker that processes CV uploads from the task queue
async def _cv_worker():
    """
    Runs the background worker that processes queued CV uploads.
    """
    from app.services.task_queue import run_cv_worker
    from app.core.db import SessionLocal
    from app.services.enhanced_cv_parser import get_enhanced_cv_parser
    from app.services.cv_parser import extract_text, parse_cv_text
    from app.services.candidate_text import upsert_candidate_embedding
    from app.services.skill_evidence import replace_candidate_skill_evidence
    from app.services.skill_status_refinement import enqueue_if_needed, refine_candidate_skill_status
    from app.models.candidate import Candidate
    from sqlalchemy import select
    from pathlib import Path
    import uuid

    # Save the uploaded CV file to disk for later download
    def _save_processed_cv(candidate_id: str, file_name: str, content: bytes) -> str:
        """
        Saves a processed CV file and returns its API URL.
        """
        storage = Path(settings.cv_storage_path)
        storage.mkdir(parents=True, exist_ok=True)
        ext = Path(file_name).suffix.lower()
        if ext not in (".pdf", ".docx", ".txt"):
            ext = ".pdf"
        dest = storage / f"{candidate_id}{ext}"
        dest.write_bytes(content)
        return f"/api/v1/candidates/{candidate_id}/cv"

    # Remove old CV files when a candidate is re-uploaded
    def _delete_processed_cv_files(candidate_id: str) -> None:
        """
        Removes old stored CV files for a candidate.
        """
        storage = Path(settings.cv_storage_path)
        for ext in (".pdf", ".docx", ".txt"):
            (storage / f"{candidate_id}{ext}").unlink(missing_ok=True)

    # Copy all parsed profile fields onto the DB candidate record
    def _apply_profile_to_candidate(candidate: Candidate, profile) -> None:
        """
        Copies parsed profile fields onto a candidate database model.
        """
        candidate.full_name = profile.full_name
        candidate.email = profile.email
        candidate.phone = profile.phone
        candidate.skills = profile.skills
        candidate.skills_detailed = [s.model_dump() if hasattr(s, "model_dump") else s for s in (profile.skills_detailed or [])]
        candidate.experience = profile.experience
        candidate.experience_entries = [e.model_dump() if hasattr(e, "model_dump") else e for e in (profile.experience_entries or [])]
        candidate.education = profile.education
        candidate.education_entries = [e.model_dump() if hasattr(e, "model_dump") else e for e in (profile.education_entries or [])]
        candidate.projects = profile.projects
        candidate.raw_text = profile.raw_text
        candidate.total_years_experience = profile.total_years_experience
        candidate.negative_skills = profile.negative_skills or None
        candidate.learning_skills = profile.learning_skills or None
        candidate.uncatalogued_skills = profile.uncatalogued_skills or None

    async def process_cv(
        cv_text: str | None,
        file_name: str,
        use_llm: bool,
        file_path: str | None = None,
        created_by_user_id: str | None = None,
    ) -> dict:
        """
        Processes one queued CV upload into a candidate record.
        """
        content: bytes | None = None
        pending_path = Path(file_path) if file_path else None
        if pending_path and pending_path.exists():
            content = pending_path.read_bytes()
            cv_text = extract_text(file_name, content)
        if not cv_text:
            raise ValueError("CV text could not be extracted")

        async with SessionLocal() as session:
            parser = get_enhanced_cv_parser() if use_llm else None
            profile = await parser.parse_async(cv_text) if parser else parse_cv_text(cv_text)

            if profile.email:
                stmt = (
                    select(Candidate)
                    .where(Candidate.email == profile.email)
                    .order_by(Candidate.id.asc())
                )
                if created_by_user_id:
                    stmt = stmt.where(Candidate.created_by_user_id == created_by_user_id)
                result = await session.execute(stmt)
                existing = result.scalars().first()
                if existing:
                    _apply_profile_to_candidate(existing, profile)
                    if content is not None:
                        existing.cv_file_name = file_name
                        existing.cv_content = content
                    await session.commit()
                    await replace_candidate_skill_evidence(session, existing, commit=True)
                    if content is not None:
                        _delete_processed_cv_files(existing.id)
                        cv_url = _save_processed_cv(existing.id, file_name, content)
                    else:
                        cv_url = f"/api/v1/candidates/{existing.id}/cv"
                    try:
                        await upsert_candidate_embedding(session, existing.id, profile)
                    except Exception:
                        logger.warning(
                            "Embedding update failed for existing candidate %s - candidate data saved",
                            existing.id,
                        )
                    if pending_path:
                        pending_path.unlink(missing_ok=True)
                    await enqueue_if_needed(existing)
                    return {
                        "candidate_id": existing.id,
                        "cv_url": cv_url,
                        "full_name": profile.full_name,
                        "email": profile.email,
                        "skills": profile.skills,
                        "uncatalogued_skills": profile.uncatalogued_skills,
                        "total_years_experience": profile.total_years_experience,
                        "status": "updated",
                    }

            candidate_id = str(uuid.uuid4())
            candidate = Candidate(
                id=candidate_id,
                created_by_user_id=created_by_user_id,
                cv_file_name=file_name if content is not None else None,
                cv_content=content,
            )
            _apply_profile_to_candidate(candidate, profile)
            session.add(candidate)
            await session.commit()
            await replace_candidate_skill_evidence(session, candidate, commit=True)

            cv_url = (
                _save_processed_cv(candidate_id, file_name, content)
                if content is not None
                else f"/api/v1/candidates/{candidate_id}/cv"
            )
            if pending_path:
                pending_path.unlink(missing_ok=True)

            try:
                await upsert_candidate_embedding(session, candidate_id, profile)
            except Exception:
                logger.warning(
                    "Embedding generation failed for candidate %s — candidate created without vector",
                    candidate_id,
                )
            await enqueue_if_needed(candidate)

            return {
                "candidate_id": candidate_id,
                "cv_url": cv_url,
                "full_name": profile.full_name,
                "email": profile.email,
                "skills": profile.skills,
                "uncatalogued_skills": profile.uncatalogued_skills,
                "total_years_experience": profile.total_years_experience,
                "status": "created",
            }

    async def refine_skill_status(candidate_id: str, raw_text_hash: str | None = None) -> dict:
        """
        Runs the background LLM pass over one candidate's unresolved skill sentences.
        """
        async with SessionLocal() as session:
            return await refine_candidate_skill_status(session, candidate_id, raw_text_hash)

    await run_cv_worker(process_cv, refine_skill_status)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Starts and stops application resources around the FastAPI lifespan.
    """
    configure_logging()
    try:
        settings.validate_runtime()
        await init_db()
        await _ensure_initial_owner()
        if settings.is_production and await get_redis() is None:
            raise RuntimeError("Redis is required for production CV task queueing")
        logger.info("Embedding provider: %s", settings.embedding_provider)
        await _warm_embedding_model()
        knowledge_task = _start_knowledge_sync()
        worker_task = asyncio.create_task(_cv_worker()) if settings.run_cv_worker_in_api else None
        keep_alive_task = asyncio.create_task(_keep_alive_worker()) if _should_run_keep_alive() else None
        if worker_task is None:
            logger.info("In-process CV worker disabled; expecting external worker service")
        yield
        for task in (worker_task, keep_alive_task, knowledge_task):
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    finally:
        await close_redis()
        try:
            from app.services.ollama_cross_encoder import get_ollama_cross_encoder

            await get_ollama_cross_encoder().close()
        except Exception:
            logger.debug("Cross-encoder cleanup skipped", exc_info=True)
        shutdown_logging()


# Tunnel domains are shared by every ngrok/localhost.run user, so trusting them with
# credentialed CORS lets any tunnel owner call the API as a logged-in user. They are
# opt-in (CORS_ALLOW_TUNNELS=true) and never honored in production.
TUNNEL_HOST_PATTERNS = ["*.ngrok-free.app", "*.ngrok-free.dev", "*.ngrok.io", "*.ngrok.app", "*.lhr.life"]
TUNNEL_ORIGIN_REGEX = r"https://([a-z0-9-]+\.)+(ngrok-free\.app|ngrok-free\.dev|ngrok\.io|ngrok\.app|lhr\.life)"
LOCAL_ORIGIN_REGEX = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"


def _tunnels_allowed() -> bool:
    """
    Checks whether tunnel domains are opted in for this non-production run.
    """
    if settings.cors_allow_tunnels and settings.is_production:
        logger.warning("CORS_ALLOW_TUNNELS is ignored in production; list tunnel origins explicitly")
        return False
    return settings.cors_allow_tunnels


def _trusted_hosts() -> list[str]:
    """
    Returns configured hosts plus loopback, and tunnel wildcards only when opted in.
    """
    hosts = settings.trusted_hosts + ["localhost", "127.0.0.1"]
    if _tunnels_allowed():
        hosts += TUNNEL_HOST_PATTERNS
    return list(dict.fromkeys(hosts))


def _cors_origin_regex() -> str | None:
    """
    Returns the extra CORS origin pattern: any localhost port in development, plus
    tunnel domains only when opted in. Production relies on CORS_ORIGINS_STR alone.
    """
    if settings.is_production:
        return None
    if _tunnels_allowed():
        return f"({LOCAL_ORIGIN_REGEX})|({TUNNEL_ORIGIN_REGEX})"
    return LOCAL_ORIGIN_REGEX


def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI application.
    """
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=_trusted_hosts())
    if settings.rate_limit_enabled:
        app.add_middleware(
            RateLimitMiddleware,
            requests=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=_cors_origin_regex(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix=settings.api_prefix)

    @app.get("/health", tags=["Health"])
    @app.get("/", tags=["Health"])
    async def root_health() -> dict[str, str]:
        """
        Root health check for load balancers and Render pings.
        """
        return {"status": "ok", "app": settings.app_name}

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        Logs unhandled errors and returns a generic API error response.
        """
        logger.exception("Unhandled application error", extra={"path": request.url.path})
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app


app = create_app()
