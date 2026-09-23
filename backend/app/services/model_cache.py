from __future__ import annotations

import os
from pathlib import Path

from app.core.config import settings

# A sentence-transformers repo is usable only when its module list and weights exist;
# a partial download leaves just the small config files behind.
SENTENCE_TRANSFORMER_REQUIRED_FILES = ("modules.json",)
SENTENCE_TRANSFORMER_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
# faster-whisper size used for local speech-to-text fallback.
WHISPER_MODEL_SIZE = "base"

_TRUE_VALUES = {"1", "true", "yes", "on"}


class ModelNotCachedError(RuntimeError):
    """Raised when a model is missing locally and runtime downloads are disabled."""


def hf_offline() -> bool:
    """
    Checks whether the HuggingFace Hub offline switch is set.
    """
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in _TRUE_VALUES


def model_downloads_allowed() -> bool:
    """
    Decides whether a missing model may be downloaded while serving requests.
    """
    if hf_offline():
        return False
    if settings.allow_model_download is not None:
        return settings.allow_model_download
    return settings.is_production


def _cached_file(repo_id: str, filename: str) -> str | None:
    """
    Returns the local path of a cached HuggingFace file, or None when it is missing.
    """
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(repo_id, filename)
    return path if isinstance(path, str) else None


def is_sentence_transformer_cached(model_name: str) -> bool:
    """
    Checks whether a sentence-transformers model has its config and weights cached.
    """
    if Path(model_name).is_dir():
        return True
    if not all(_cached_file(model_name, name) for name in SENTENCE_TRANSFORMER_REQUIRED_FILES):
        return False
    return any(_cached_file(model_name, name) for name in SENTENCE_TRANSFORMER_WEIGHT_FILES)


def missing_model_message(model_name: str) -> str:
    """
    Builds the log/readiness message for a model that is not available locally.
    """
    return (
        f"Model '{model_name}' is not fully cached (weights missing) and runtime downloads "
        "are disabled. Run `python scripts/download_models.py` from backend/, "
        "or set ALLOW_MODEL_DOWNLOAD=true."
    )
