from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

_redis = None
# After a failed connect, skip retries briefly: a refused connection can take ~2s on
# Windows, and every caller would otherwise pay it.
_UNAVAILABLE_RETRY_SECONDS = 5.0
_unavailable_until = 0.0


async def get_redis():
    """
    Returns a Redis client when Redis is available.
    """
    global _redis, _unavailable_until
    if _redis is None and time.monotonic() >= _unavailable_until:
        try:
            import redis.asyncio as aioredis
            client = aioredis.from_url(str(settings.redis_url), decode_responses=True)
            await client.ping()
            # Publish only a verified client; concurrent callers must not get one
            # whose ping is still pending (or about to fail).
            _redis = client
            logger.info("Connected to Redis")
        except Exception:
            logger.warning("Redis not available, caching disabled")
            _unavailable_until = time.monotonic() + _UNAVAILABLE_RETRY_SECONDS
    return _redis


async def cache_get(key: str) -> Any | None:
    """
    Reads and JSON-decodes a value from Redis cache.
    """
    r = await get_redis()
    if r is None:
        return None
    try:
        data = await r.get(key)
        return json.loads(data) if data else None
    except Exception:
        return None


async def cache_set(key: str, value: Any, ttl: int = 300) -> None:
    """
    JSON-encodes and stores a value in Redis cache.
    """
    r = await get_redis()
    if r is None:
        return
    try:
        await r.setex(key, ttl, json.dumps(value))
    except Exception:
        pass


async def cache_delete(key: str) -> None:
    """
    Deletes a value from Redis cache.
    """
    r = await get_redis()
    if r is None:
        return
    try:
        await r.delete(key)
    except Exception:
        pass


async def close_redis() -> None:
    """
    Closes the cached Redis connection.
    """
    global _redis
    if _redis:
        close_func = getattr(_redis, "aclose", None) or getattr(_redis, "close", None)
        if close_func:
            result = close_func()
            if hasattr(result, "__await__"):
                await result
        _redis = None
