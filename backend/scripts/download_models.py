"""
Pre-downloads every HuggingFace model the backend can load and verifies the weights.

Run from backend/ before a demo or in a build step:

    python scripts/download_models.py            # embeddings + faster-whisper
    python scripts/download_models.py --skip-whisper

At runtime, outside production, a model that is not fully cached fails fast and the
caller falls back instead of downloading mid-request (see ALLOW_MODEL_DOWNLOAD).
Exits non-zero if any model is still missing or fails to load.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app  # noqa: E402,F401  (sets USE_TF=0 before transformers is imported)
from app.core.config import settings  # noqa: E402
from app.services.model_cache import (  # noqa: E402
    SENTENCE_TRANSFORMER_WEIGHT_FILES,
    WHISPER_MODEL_SIZE,
    is_sentence_transformer_cached,
)

# Alternate weight formats the PyTorch loader never reads; skipping them keeps the
# multilingual download to ~0.5 GB instead of several GB.
IGNORED_PATTERNS = [
    "onnx/*", "openvino/*", "*.onnx", "*.h5", "*.msgpack", "*.ot",
    "tf_model*", "flax_model*", "rust_model*",
]


def sentence_transformer_models() -> list[str]:
    """
    Returns the sentence-transformers models the current config can route to.
    """
    models = [settings.embedding_model]
    # Arabic text is routed to the multilingual model when either flag is on (default: auto_detect_lang).
    if settings.auto_detect_lang or settings.use_multilingual_embedding:
        models.append(settings.multilingual_embedding_model)
    return list(dict.fromkeys(models))


def download_sentence_transformer(model_name: str) -> None:
    """
    Downloads one sentence-transformers repo, preferring safetensors weights.
    """
    from huggingface_hub import HfApi, snapshot_download

    ignore = list(IGNORED_PATTERNS)
    files = set(HfApi().list_repo_files(model_name))
    if SENTENCE_TRANSFORMER_WEIGHT_FILES[0] in files:
        ignore.append("pytorch_model.bin")
    snapshot_download(model_name, ignore_patterns=ignore)


def verify_sentence_transformer(model_name: str) -> None:
    """
    Loads a cached model offline and encodes one sentence to prove the weights work.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, local_files_only=True)
    vector = model.encode(["python backend developer"], normalize_embeddings=True)[0]
    print(f"    loaded offline, dimension={len(vector)}")


def download_whisper() -> None:
    """
    Downloads the faster-whisper model used for local speech-to-text.
    """
    from faster_whisper import download_model

    download_model(WHISPER_MODEL_SIZE)


def verify_whisper() -> None:
    """
    Loads faster-whisper from the local cache only.
    """
    from faster_whisper import WhisperModel

    WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8", local_files_only=True)
    print("    loaded offline")


def _run(label: str, download, verify, is_cached=None, skip_load: bool = False) -> bool:
    """
    Downloads and verifies one model, returning True on success.
    """
    print(f"==> {label}")
    started = time.monotonic()
    try:
        if is_cached is not None and is_cached():
            print("    already cached")
        else:
            print("    downloading...")
            download()
        if is_cached is not None and not is_cached():
            print("    FAILED: weights are still missing from the HuggingFace cache")
            return False
        if not skip_load:
            verify()
    except Exception as exc:
        print(f"    FAILED: {type(exc).__name__}: {exc}")
        print(
            "    Hint: if a previous download was interrupted, stop other Python processes "
            "and delete the model's *.incomplete blob and its ~/.cache/huggingface/hub/.locks entry."
        )
        return False
    print(f"    ok ({time.monotonic() - started:.1f}s)")
    return True


def main() -> int:
    """
    Downloads and verifies all configured models.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-whisper", action="store_true", help="Skip the faster-whisper STT model.")
    parser.add_argument("--no-load-check", action="store_true", help="Only check files; do not load models.")
    args = parser.parse_args()

    results = []
    for model_name in sentence_transformer_models():
        results.append(_run(
            model_name,
            lambda m=model_name: download_sentence_transformer(m),
            lambda m=model_name: verify_sentence_transformer(m),
            is_cached=lambda m=model_name: is_sentence_transformer_cached(m),
            skip_load=args.no_load_check,
        ))
    if not args.skip_whisper:
        results.append(_run(
            f"faster-whisper-{WHISPER_MODEL_SIZE}",
            download_whisper,
            verify_whisper,
            skip_load=args.no_load_check,
        ))

    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} models ready")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
