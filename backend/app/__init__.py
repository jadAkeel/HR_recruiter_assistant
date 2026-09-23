import os

# Must run before anything imports transformers. stanza (pulled in by cv_parser)
# imports it at module load, so setting these later - or in .env, which
# pydantic-settings never exports to os.environ - is too late. Without them
# transformers tries to load TensorFlow/Keras 3, fails, and embeddings silently
# fall back to hash vectors. This package init is the earliest shared point for
# the API, the standalone worker, scripts, and tests.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
