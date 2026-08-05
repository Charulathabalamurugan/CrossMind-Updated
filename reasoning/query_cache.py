import time
import threading
import logging
import json
from typing import Any, Optional, Dict
from config import settings
from ingestion.embedding import get_embedder

logger = logging.getLogger("crossmind.query_cache")

class QueryResultCache:
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._timestamps: Dict[str, float] = {}
        self._query_embeddings: Dict[str, list] = {}
        self._lock = threading.Lock()
        self.max_items = settings.INGESTION_CACHE_MAX_ITEMS
        self.ttl = settings.INGESTION_CACHE_TTL_SECONDS

    def _evict_expired(self):
        now = time.time()
        expired = [k for k, ts in self._timestamps.items() if now - ts > self.ttl]
        for key in expired:
            self._store.pop(key, None)
            self._timestamps.pop(key, None)

    def _normalize_key(self, key: Any) -> str:
        try:
            return json.dumps(key, sort_keys=True)
        except Exception:
            return str(key)

    def _evict_lru(self):
        if len(self._store) > self.max_items:
            sorted_items = sorted(self._timestamps.items(), key=lambda x: x[1])
            remove_count = len(self._store) - self.max_items
            for key, _ in sorted_items[:remove_count]:
                self._store.pop(key, None)
                self._timestamps.pop(key, None)
                self._query_embeddings.pop(key, None)

    def _query_embedding(self, query: str) -> Optional[list]:
        try:
            embedder = get_embedder()
            return embedder.embed_text(query, dim=settings.SEMANTIC_QUERY_CACHE_DIM)
        except Exception:
            return None

    def _cosine_similarity(self, a: list, b: list) -> float:
        try:
            import numpy as np
            a_arr = np.asarray(a, dtype=np.float32)
            b_arr = np.asarray(b, dtype=np.float32)
            if a_arr.size == 0 or b_arr.size == 0:
                return 0.0
            dot = float(np.dot(a_arr, b_arr))
            norm = float(np.linalg.norm(a_arr)) * float(np.linalg.norm(b_arr))
            return dot / norm if norm > 0 else 0.0
        except Exception:
            return 0.0

    def get(self, key: Any) -> Optional[Any]:
        normalized_key = self._normalize_key(key)
        with self._lock:
            self._evict_expired()
            entry = self._store.get(normalized_key)
            if entry:
                return entry["result"]
            return None

    def get_similar(self, query: str, user_role: Optional[str] = None, threshold: float = None) -> Optional[Any]:
        if threshold is None:
            threshold = settings.SEMANTIC_QUERY_CACHE_THRESHOLD
        query_emb = self._query_embedding(query)
        if query_emb is None:
            return None

        with self._lock:
            self._evict_expired()
            best_match = None
            best_score = threshold
            for key, emb in self._query_embeddings.items():
                meta = self._store.get(key, {})
                if user_role and meta.get("user_role") != user_role:
                    continue
                similarity = self._cosine_similarity(query_emb, emb)
                if similarity > best_score:
                    best_score = similarity
                    best_match = meta.get("result")
            return best_match

    def set(self, key: Any, result: Any, query: Optional[str] = None, user_role: Optional[str] = None):
        normalized_key = self._normalize_key(key)
        with self._lock:
            self._evict_expired()
            entry = {"result": result, "created": time.time(), "query": query, "user_role": user_role}
            self._store[normalized_key] = entry
            self._timestamps[normalized_key] = time.time()
            if query:
                query_emb = self._query_embedding(query)
                if query_emb is not None:
                    self._query_embeddings[normalized_key] = query_emb
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

_query_cache_instance = None

def get_query_cache() -> QueryResultCache:
    global _query_cache_instance
    if _query_cache_instance is None:
        _query_cache_instance = QueryResultCache()
    return _query_cache_instance
