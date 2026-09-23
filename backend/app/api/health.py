from __future__ import annotations

import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.schemas.health import HealthResponse
from app.services.readiness import collect_readiness

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Returns the basic API health status.
    """
    return HealthResponse(status="ok")


@router.get("/ready")
async def readiness() -> JSONResponse:
    """
    Checks runtime readiness for production dependencies.
    """
    body = await collect_readiness()
    status_code = 200 if body["status"] == "ok" else 503
    return JSONResponse(status_code=status_code, content=body)


@router.get("/health/embeddings")
async def health_embeddings():
    """
    Returns embedding provider status and checks if embeddings are real or mock fallbacks.
    """
    from app.core.config import settings
    from app.services.embedding import embedding_runtime_status

    provider = settings.embedding_provider.lower()
    # Only models that have been loaded appear here; any of them on the hash fallback
    # means vectors are not real even though a real provider is configured.
    runtime = {
        model: status
        for model, status in embedding_runtime_status().items()
        if status["configured_provider"] == provider
    }
    fallback_models = sorted(model for model, status in runtime.items() if status["active_provider"] != provider)
    is_real = provider != "hash" and not fallback_models
    return {
        "status": "ok",
        "provider": provider,
        "is_real": is_real,
        "model_name": settings.embedding_model,
        "multilingual_model_name": settings.multilingual_embedding_model,
        "dimension": settings.embedding_dimension,
        "use_multilingual_embedding": settings.use_multilingual_embedding,
        "auto_detect_lang": settings.auto_detect_lang,
        "is_fallback": not is_real,
        "fallback_models": fallback_models,
        "loaded_models": runtime,
    }

