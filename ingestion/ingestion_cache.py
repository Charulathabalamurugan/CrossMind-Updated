import time
import threading
import logging
from typing import Any, Optional
from config import settings

logger = logging.getLogger("crossmind.ingestion_cache")

class IngestionCache:
    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.Lock()
        self.max_items = settings.INGESTION_CACHE_MAX_ITEMS
        self.ttl = settings.INGESTION_CACHE_TTL_SECONDS

    def _evict_expired(self):
        now = time.time()
        expired = [k for k, ts in self._timestamps.items() if now - ts > self.ttl]
        for key in expired:
            self._store.pop(key, None)
            self._timestamps.pop(key, None)

    def _evict_lru(self):
        if len(self._store) > self.max_items:
            sorted_items = sorted(self._timestamps.items(), key=lambda x: x[1])
            remove_count = len(self._store) - self.max_items
            for key, _ in sorted_items[:remove_count]:
                self._store.pop(key, None)
                self._timestamps.pop(key, None)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            self._evict_expired()
            return self._store.get(key)

    def set(self, key: str, value: Any):
        with self._lock:
            self._evict_expired()
            self._store[key] = value
            self._timestamps[key] = time.time()
            self._evict_lru()

    def delete(self, key: str):
        with self._lock:
            self._store.pop(key, None)
            self._timestamps.pop(key, None)

    def clear(self):
        with self._lock:
            self._store.clear()
            self._timestamps.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._store)

_cache_instance: Optional[IngestionCache] = None

def get_ingestion_cache() -> IngestionCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = IngestionCache()
    return _cache_instance
