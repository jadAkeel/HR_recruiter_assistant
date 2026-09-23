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
# Warn once per outage; later retries would repeat it every few seconds.
_outage_logged = False


async def get_redis():
    """
    Returns a Redis client when Redis is available.
    """
    global _redis, _unavailable_until, _outage_logged
    if _redis is None and time.monotonic() >= _unavailable_until:
        client = None
        try:
            import redis.asyncio as aioredis
            client = aioredis.from_url(str(settings.redis_url), decode_responses=True)
            await client.ping()
            # Publish only a verified client; concurrent callers must not get one
            # whose ping is still pending (or about to fail).
            _redis = client
            _outage_logged = False
            logger.info("Connected to Redis")
        except Exception:
            if client is not None:
                await _close_client(client)
            if _outage_logged:
                logger.debug("Redis still not available")
            else:
                logger.warning("Redis not available, caching disabled")
                _outage_logged = True
            _unavailable_until = time.monotonic() + _UNAVAILABLE_RETRY_SECONDS
    return _redis


async def _close_client(client) -> None:
    """
    Closes a Redis client, ignoring errors from a connection that never opened.
    """
    try:
        close_func = getattr(client, "aclose", None) or getattr(client, "close", None)
        if close_func:
            result = close_func()
            if hasattr(result, "__await__"):
                await result
    except Exception:
        logger.debug("Redis client close failed", exc_info=True)


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
        await _close_client(_redis)
        _redis = None
