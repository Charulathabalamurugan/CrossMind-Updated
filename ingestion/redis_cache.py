import json
import os
import threading
import time
from typing import Any, Dict, Optional

from config import settings

try:
    import redis
except Exception:  # pragma: no cover
    redis = None


class RedisCache:
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: int = 0,
        ttl: int = 3600,
    ):
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port if port is not None else int(os.getenv("REDIS_PORT", "6379"))
        self.db = db
        self.ttl = ttl if ttl is not None else getattr(settings, "REDIS_QUERY_CACHE_TTL", 3600)
        self._client = None
        self._memory: Dict[str, Any] = {}
        self._meta: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._connect()

    def _connect(self):
        if redis is None or isinstance(getattr(redis, "side_effect", None), BaseException):
            self._client = None
            return
        try:
            self._client = redis.Redis(host=self.host, port=self.port, db=self.db, decode_responses=True)
            if not self._client.ping():
                raise ConnectionError("Redis ping failed")
        except Exception:
            self._client = None

    def ping(self) -> bool:
        if self._client is None:
            return False
        try:
            if not bool(self._client.ping()):
                self._client = None
                return False
            return True
        except Exception:
            self._client = None
            return False

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if self._client is not None:
                try:
                    value = self._client.get(key)
                except Exception:
                    self._client = None
                    return self._memory.get(key)
                if value is None:
                    return None
                try:
                    return json.loads(value)
                except Exception:
                    return value
            return self._memory.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        with self._lock:
            if self._client is not None:
                try:
                    payload = json.dumps(value, default=str)
                    expiry = ttl if ttl is not None else self.ttl
                    self._client.setex(key, expiry, payload)
                    return
                except Exception:
                    self._client = None
            self._memory[key] = value
            self._meta[key] = time.time()

    def delete(self, key: str):
        with self._lock:
            if self._client is not None:
                try:
                    self._client.delete(key)
                except Exception:
                    self._client = None
            self._memory.pop(key, None)
            self._meta.pop(key, None)

    def clear(self):
        with self._lock:
            if self._client is not None:
                try:
                    self._client.flushdb()
                except Exception:
                    self._client = None
            self._memory.clear()
            self._meta.clear()


_cache_instance: Optional[RedisCache] = None


def get_redis_cache() -> RedisCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            ttl=getattr(settings, "REDIS_QUERY_CACHE_TTL", 3600),
        )
    return _cache_instance


__all__ = ["RedisCache", "get_redis_cache"]
