import time
import logging
import json
from typing import Optional, Any
from config import settings

logger = logging.getLogger("crossmind.redis_cache")

try:
    import redis
    REDIS_AVAILABLE = True
    _redis_client = redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        decode_responses=True,
    )
    _redis_client.ping()
    logger.info("Redis connection established for dedup cache.")
except Exception:
    REDIS_AVAILABLE = False
    _redis_client = None
    logger.info("Redis not available. Using in-memory dedup cache.")

import os

class DedupCache:
    def __init__(self):
        self._memory_cache: dict = {}
        self.ttl = settings.INGESTION_CACHE_TTL_SECONDS

    def is_seen(self, content_hash: str) -> bool:
        if REDIS_AVAILABLE and _redis_client:
            try:
                return bool(_redis_client.exists(f"crossmind:hash:{content_hash}"))
            except Exception:
                pass
        return content_hash in self._memory_cache

    def mark_seen(self, content_hash: str):
        if REDIS_AVAILABLE and _redis_client:
            try:
                _redis_client.setex(
                    f"crossmind:hash:{content_hash}",
                    self.ttl,
                    "1",
                )
                return
            except Exception:
                pass
        self._memory_cache[content_hash] = time.time()
        self._evict_expired()

    def _evict_expired(self):
        now = time.time()
        expired = [k for k, t in self._memory_cache.items() if now - t > self.ttl]
        for k in expired:
            self._memory_cache.pop(k, None)

    def flush(self):
        if REDIS_AVAILABLE and _redis_client:
            try:
                _redis_client.flushdb()
            except Exception:
                pass
        self._memory_cache.clear()

    def size(self) -> int:
        if REDIS_AVAILABLE and _redis_client:
            try:
                return int(_redis_client.db_size())
            except Exception:
                pass
        return len(self._memory_cache)

_dedup_cache_instance = None

def get_dedup_cache() -> DedupCache:
    global _dedup_cache_instance
    if _dedup_cache_instance is None:
        _dedup_cache_instance = DedupCache()
    return _dedup_cache_instance