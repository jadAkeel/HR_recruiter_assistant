import asyncio
import sys
import types

import pytest

from app.services import embedding, model_cache, voice_service


@pytest.fixture
def fake_hf_cache(monkeypatch):
    """
    Replaces the HuggingFace cache lookup with an in-memory set of cached files.
    """
    cached: set[tuple[str, str]] = set()
    monkeypatch.setattr(
        model_cache,
        "_cached_file",
        lambda repo_id, filename: f"/cache/{repo_id}/{filename}" if (repo_id, filename) in cached else None,
    )
    return cached


@pytest.fixture
def forbid_sentence_transformers_import(monkeypatch):
    """
    Makes any attempt to construct a SentenceTransformer fail the test.
    """
    module = types.ModuleType("sentence_transformers")

    def _forbidden(*args, **kwargs):
        raise AssertionError("SentenceTransformer must not be constructed for an uncached model")

    module.SentenceTransformer = _forbidden
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_partial_cache_without_weights_is_not_cached(fake_hf_cache) -> None:
    """
    Checks that config-only leftovers of an interrupted download do not count as cached.
    """
    model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    fake_hf_cache.update({(model, "modules.json"), (model, "config.json")})
    assert model_cache.is_sentence_transformer_cached(model) is False

    fake_hf_cache.add((model, "model.safetensors"))
    assert model_cache.is_sentence_transformer_cached(model) is True


def test_downloads_disabled_outside_production_and_when_offline(monkeypatch) -> None:
    """
    Checks the runtime download policy: dev fails fast, production downloads, and
    HF_HUB_OFFLINE always wins.
    """
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(model_cache.settings, "allow_model_download", None)
    monkeypatch.setattr(model_cache.settings, "environment", "development")
    assert model_cache.model_downloads_allowed() is False

    monkeypatch.setattr(model_cache.settings, "environment", "production")
    assert model_cache.model_downloads_allowed() is True

    monkeypatch.setattr(model_cache.settings, "environment", "development")
    monkeypatch.setattr(model_cache.settings, "allow_model_download", True)
    assert model_cache.model_downloads_allowed() is True

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert model_cache.model_downloads_allowed() is False


def test_uncached_model_fails_fast_without_downloading(
    monkeypatch, fake_hf_cache, forbid_sentence_transformers_import
) -> None:
    """
    Checks that a missing model raises a clear error instead of starting a download.
    """
    monkeypatch.setattr(model_cache.settings, "allow_model_download", False)

    with pytest.raises(model_cache.ModelNotCachedError, match="download_models.py"):
        embedding.LocalEmbeddingService("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def test_cached_model_loads_with_local_files_only(monkeypatch, fake_hf_cache) -> None:
    """
    Checks that a fully cached model is loaded without any HuggingFace network call.
    """
    model = "sentence-transformers/all-MiniLM-L6-v2"
    fake_hf_cache.update({(model, "modules.json"), (model, "model.safetensors")})
    calls = []
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = lambda name, **kwargs: calls.append((name, kwargs)) or object()
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    embedding.LocalEmbeddingService(model)

    assert calls == [(model, {"local_files_only": True})]


def test_arabic_text_falls_back_when_multilingual_model_is_missing(
    monkeypatch, fake_hf_cache, forbid_sentence_transformers_import
) -> None:
    """
    Checks that an uncached multilingual model degrades Arabic text to the fallback
    instead of blocking or failing the request.
    """
    monkeypatch.setattr(embedding.settings, "embedding_provider", "sentence-transformers")
    monkeypatch.setattr(embedding.settings, "auto_detect_lang", True)
    monkeypatch.setattr(embedding.settings, "use_multilingual_embedding", False)
    monkeypatch.setattr(model_cache.settings, "allow_model_download", False)
    monkeypatch.setattr(embedding.settings, "multilingual_embedding_dimension", 768)
    fake_hf_cache.update({
        (embedding.settings.embedding_model, "modules.json"),
        (embedding.settings.embedding_model, "model.safetensors"),
    })

    class StandardModel:
        def __init__(self, name, **kwargs):
            """
            Initializes a fake standard model.
            """

        def encode(self, texts, normalize_embeddings=True):
            """
            Returns fixed 384-dimensional vectors.
            """
            import numpy as np

            return np.ones((len(texts), 384)) / np.sqrt(384)

    sys.modules["sentence_transformers"].SentenceTransformer = lambda name, **kwargs: (
        StandardModel(name) if name == embedding.settings.embedding_model
        else (_ for _ in ()).throw(AssertionError("multilingual model must not be constructed"))
    )
    embedding.get_embedding_service.cache_clear()
    try:
        service = embedding.get_embedding_service()
        vectors = asyncio.run(service.embed(["python developer", "مهندس بايثون"]))
        status = embedding.embedding_runtime_status()
    finally:
        embedding.get_embedding_service.cache_clear()

    assert [len(v) for v in vectors] == [384, 768]
    assert status[embedding.settings.multilingual_embedding_model]["active_provider"] == "hash"
    assert status[embedding.settings.embedding_model]["active_provider"] == "sentence-transformers"


def test_english_only_traffic_never_builds_the_multilingual_model(monkeypatch) -> None:
    """
    Checks that the Arabic route is built lazily, so English requests never load it.
    """
    built = []

    class Provider:
        def __init__(self, dim):
            """
            Initializes a fake provider with a fixed dimension.
            """
            self.dim = dim

        async def embed(self, texts):
            """
            Returns zero vectors of the fake dimension.
            """
            return [[0.0] * self.dim for _ in texts]

    monkeypatch.setattr(embedding.settings, "auto_detect_lang", True)
    monkeypatch.setattr(embedding.settings, "use_multilingual_embedding", False)
    service = embedding.LanguageRoutingEmbeddingService(Provider(384), lambda: built.append("multi") or Provider(768))

    asyncio.run(service.embed(["python developer"]))
    assert built == []

    vectors = asyncio.run(service.embed(["مهندس بايثون", "مطور"]))
    assert built == ["multi"]
    assert [len(v) for v in vectors] == [768, 768]


def test_whisper_fails_fast_when_not_cached(monkeypatch) -> None:
    """
    Checks that faster-whisper does not download mid-request when downloads are off.
    """
    calls = []

    def fake_whisper(size, **kwargs):
        calls.append(kwargs.get("local_files_only", False))
        raise FileNotFoundError("not cached")

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = fake_whisper
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setattr(model_cache.settings, "allow_model_download", False)

    with pytest.raises(model_cache.ModelNotCachedError):
        voice_service._load_whisper_model()
    assert calls == [True]

