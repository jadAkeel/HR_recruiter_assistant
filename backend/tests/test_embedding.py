import asyncio

from app.services import embedding


def test_local_embedding_constructor_failure_uses_hash_fallback(monkeypatch) -> None:
    """
    Checks that local embedding constructor failure uses hash fallback.
    """
    class BrokenLocalEmbeddingService:
        def __init__(self, model_name: str) -> None:
            """
            Initializes a test double used by the surrounding test.
            """
            raise ValueError("local model import failed")

    monkeypatch.setattr(embedding.settings, "embedding_provider", "sentence-transformers")
    monkeypatch.setattr(embedding, "LocalEmbeddingService", BrokenLocalEmbeddingService)
    embedding.get_embedding_service.cache_clear()

    try:
        service = embedding.get_embedding_service()
        vectors = asyncio.run(service.embed(["python backend engineer"]))
    finally:
        embedding.get_embedding_service.cache_clear()

    assert len(vectors) == 1
    assert len(vectors[0]) == embedding.settings.embedding_dimension
    assert any(value != 0 for value in vectors[0])


def test_app_package_disables_tensorflow_before_anything_imports_transformers() -> None:
    """
    Checks that importing the app package sets USE_TF/TRANSFORMERS_NO_TF without any
    help from the shell, since stanza imports transformers at module load.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    env = {k: v for k, v in os.environ.items() if k not in {"USE_TF", "TRANSFORMERS_NO_TF"}}
    result = subprocess.run(
        [sys.executable, "-c", "import os, app; print(os.environ['USE_TF'], os.environ['TRANSFORMERS_NO_TF'])"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["0", "1"]


class _WorkingLocalEmbeddingService:
    def __init__(self, model_name: str) -> None:
        """
        Initializes a test double that behaves like a loaded local model.
        """
        self.model_name = model_name

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Returns fixed unit vectors with the configured dimension.
        """
        dim = embedding.settings.embedding_dimension
        return [[1.0] + [0.0] * (dim - 1) for _ in texts]


def _use_local_provider(monkeypatch, local_service) -> None:
    """
    Configures sentence-transformers with a test double and no Arabic routing.
    """
    monkeypatch.setattr(embedding.settings, "embedding_provider", "sentence-transformers")
    monkeypatch.setattr(embedding.settings, "auto_detect_lang", False)
    monkeypatch.setattr(embedding.settings, "use_multilingual_embedding", False)
    monkeypatch.setattr(embedding, "LocalEmbeddingService", local_service)
    embedding.get_embedding_service.cache_clear()


def test_readiness_reports_degraded_when_local_model_fails_to_load(monkeypatch) -> None:
    """
    Checks that readiness names the hash fallback instead of reporting the configured
    provider as reachable when the local model could not be loaded.
    """
    from app.services import readiness

    class BrokenLocalEmbeddingService:
        def __init__(self, model_name: str) -> None:
            """
            Initializes a test double whose model import fails.
            """
            raise ValueError("Keras 3 is not supported in Transformers")

    _use_local_provider(monkeypatch, BrokenLocalEmbeddingService)
    try:
        result = asyncio.run(readiness._check_embedding_provider())
    finally:
        embedding.get_embedding_service.cache_clear()

    model = embedding.settings.embedding_model
    assert result["status"] == "degraded"
    assert result["fallback_provider"] == "hash"
    assert result["fallback_models"][model]["active_provider"] == "hash"
    assert "Keras 3" in result["fallback_models"][model]["fallback_reason"]
    assert "hash fallback" in result["message"]


def test_readiness_reports_degraded_when_local_model_fails_at_runtime(monkeypatch) -> None:
    """
    Checks that a model that loads but then fails to embed is reported as degraded.
    """
    from app.services import readiness

    class FailingLocalEmbeddingService(_WorkingLocalEmbeddingService):
        async def embed(self, texts: list[str]) -> list[list[float]]:
            """
            Raises like a model that crashes during encoding.
            """
            raise RuntimeError("encode failed")

    _use_local_provider(monkeypatch, FailingLocalEmbeddingService)
    try:
        result = asyncio.run(readiness._check_embedding_provider())
    finally:
        embedding.get_embedding_service.cache_clear()

    assert result["status"] == "degraded"
    assert embedding.settings.embedding_model in result["fallback_models"]


def test_readiness_reports_ok_when_local_model_serves_embeddings(monkeypatch) -> None:
    """
    Checks that readiness reports the real provider when the local model works.
    """
    from app.services import readiness

    _use_local_provider(monkeypatch, _WorkingLocalEmbeddingService)
    try:
        result = asyncio.run(readiness._check_embedding_provider())
    finally:
        embedding.get_embedding_service.cache_clear()

    assert result["status"] == "ok"
    assert result["active_provider"] == "sentence-transformers"
    assert "fallback_models" not in result


def test_embedding_metadata_labels_fallback_vectors_as_hash(monkeypatch) -> None:
    """
    Checks that vectors produced by the hash fallback are stored as hash, so they are
    not reused as semantic vectors after the real model loads again.
    """
    class BrokenLocalEmbeddingService:
        def __init__(self, model_name: str) -> None:
            """
            Initializes a test double whose model import fails.
            """
            raise ValueError("local model import failed")

    _use_local_provider(monkeypatch, BrokenLocalEmbeddingService)
    try:
        asyncio.run(embedding.get_embedding_service().embed(["python backend engineer"]))
        fallback_metadata = embedding.embedding_metadata_for_text("python backend engineer")
    finally:
        embedding.get_embedding_service.cache_clear()
    healthy_metadata = embedding.embedding_metadata_for_text("python backend engineer")

    assert fallback_metadata["provider"] == "hash"
    assert fallback_metadata["is_fallback"] is True
    assert healthy_metadata["provider"] == "sentence-transformers"
    assert healthy_metadata["is_fallback"] is False


def test_health_embeddings_reports_fallback(monkeypatch) -> None:
    """
    Checks that /health/embeddings stops claiming real vectors once a model fell back.
    """
    from fastapi.testclient import TestClient

    from app.main import create_app

    class BrokenLocalEmbeddingService:
        def __init__(self, model_name: str) -> None:
            """
            Initializes a test double whose model import fails.
            """
            raise ValueError("local model import failed")

    monkeypatch.setattr(embedding.settings, "embedding_provider", "sentence-transformers")
    monkeypatch.setattr(embedding.settings, "auto_detect_lang", False)
    monkeypatch.setattr(embedding, "LocalEmbeddingService", BrokenLocalEmbeddingService)
    embedding.get_embedding_service.cache_clear()
    try:
        embedding.get_embedding_service()
        body = TestClient(create_app()).get("/api/v1/health/embeddings").json()
    finally:
        embedding.get_embedding_service.cache_clear()

    assert body["is_real"] is False
    assert body["is_fallback"] is True
    assert body["fallback_models"] == [embedding.settings.embedding_model]


def test_hash_fallback_vectors_are_detectable_from_source_hash() -> None:
    """
    Checks that a stored hash-fallback vector can be recognised from its source hash,
    so vectors mislabelled as real are recomputed instead of reused.
    """
    import hashlib

    text = "python backend engineer with fastapi"
    fallback = embedding.DevelopmentFallbackEmbedding(dimension=384)._embed_sync([text])[0]
    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    assert embedding.is_hash_fallback_vector(fallback, source_hash) is True
    assert embedding.is_hash_fallback_vector([1.0] + [0.0] * 383, source_hash) is False
    assert embedding.is_hash_fallback_vector(fallback, None) is False


def test_matcher_ignores_cached_hash_vector_labelled_as_real(monkeypatch) -> None:
    """
    Checks that a candidate vector saved by the silent fallback under a real provider
    label is not reused for semantic scoring.
    """
    from types import SimpleNamespace

    from app.services import hybrid_matcher

    text = "python backend engineer with fastapi"
    fallback = embedding.DevelopmentFallbackEmbedding(dimension=384)._embed_sync([text])[0]
    metadata = embedding.embedding_metadata_for_text(text, provider="sentence-transformers")
    row = SimpleNamespace(
        provider="sentence-transformers",
        model_name=metadata["model_name"],
        source_hash=metadata["source_hash"],
        embedding_json=fallback,
    )

    class FakeSession:
        async def execute(self, stmt):
            """
            Returns the stored embedding row for any query.
            """
            return SimpleNamespace(scalar_one_or_none=lambda: row)

    matcher = hybrid_matcher.HybridMatchingEngine.__new__(hybrid_matcher.HybridMatchingEngine)
    monkeypatch.setattr(matcher, "_build_candidate_text", lambda candidate: text, raising=False)
    monkeypatch.setattr(
        hybrid_matcher,
        "embedding_metadata_for_text",
        lambda value: embedding.embedding_metadata_for_text(value, provider="sentence-transformers"),
    )
    store = SimpleNamespace(session=FakeSession())

    assert asyncio.run(matcher._get_cached_embedding(store, SimpleNamespace(id="candidate-1"))) is None

    row.embedding_json = [1.0] + [0.0] * 383
    assert asyncio.run(matcher._get_cached_embedding(store, SimpleNamespace(id="candidate-1"))) == row.embedding_json


def test_multilingual_dimension_matches_configured_model() -> None:
    """
    Checks that the Arabic route uses the multilingual model's real dimension, which
    for paraphrase-multilingual-MiniLM-L12-v2 is 384 (the pgvector column size).
    """
    assert embedding.settings.multilingual_embedding_dimension == 384
    assert embedding.embedding_dimension_for_text("مهندس بايثون") == embedding.settings.multilingual_embedding_dimension
    vector = embedding.DevelopmentFallbackEmbedding()._embed_sync(["مهندس بايثون"])[0]
    assert len(vector) == embedding.settings.multilingual_embedding_dimension
