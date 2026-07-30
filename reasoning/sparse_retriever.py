import time
import logging
import hashlib
from typing import List, Dict, Any, Optional
from config import settings
from ingestion.sparse_vector import get_sparse_vector_engine

logger = logging.getLogger("crossmind.sparse_retriever")

class SparseRetriever:
    def __init__(self):
        self.sparse_engine = get_sparse_vector_engine()
        self.rrf_enabled = getattr(settings, "RRF_ENABLED", True)
        self.rrf_k = getattr(settings, "RRF_K", 60)
        self.cache_ttl = getattr(settings, "REDIS_RETRIEVAL_CACHE_TTL", 1800)
        self.cache_max = getattr(settings, "REDIS_RETRIEVAL_CACHE_MAX", 10000)
        self._redis_cache = None
        self._init_redis_cache()

    def _init_redis_cache(self):
        try:
            import redis
            self._redis_cache = redis.Redis(host='localhost', port=6379, db=2, decode_responses=True)
            self._redis_cache.ping()
            logger.info("Redis retrieval cache initialized.")
        except Exception:
            self._redis_cache = None
            logger.info("Redis retrieval cache unavailable. Running without cache.")

    def index_documents(self, documents: List[Dict[str, Any]]):
        self.sparse_engine.index_documents(documents)

    def _cache_key(self, query: str, top_k: int, mode: str) -> str:
        raw = f"{query}:{top_k}:{mode}:{self.rrf_k}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _cached_search(self, query: str, top_k: int) -> Optional[List[Dict[str, Any]]]:
        if not self._redis_cache:
            return None
        key = self._cache_key(query, top_k, "sparse")
        try:
            cached = self._redis_cache.get(key)
            if cached:
                import json
                return json.loads(cached)
        except Exception:
            pass
        return None

    def _store_cache(self, query: str, top_k: int, results: List[Dict[str, Any]]):
        if not self._redis_cache:
            return
        key = self._cache_key(query, top_k, "sparse")
        try:
            import json
            self._redis_cache.setex(key, self.cache_ttl, json.dumps(results))
        except Exception:
            pass

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        cached = self._cached_search(query, top_k)
        if cached is not None:
            return cached
        results = self.sparse_engine.search(query, top_k=top_k)
        for r in results:
            r["source"] = "sparse_tfidf"
        self._store_cache(query, top_k, results)
        return results

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        score_map = {}
        for rank, doc in enumerate(dense_results, 1):
            doc_id = str(doc.get("id", ""))
            score_map[doc_id] = score_map.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, doc in enumerate(sparse_results, 1):
            doc_id = str(doc.get("doc_id", ""))
            score_map[doc_id] = score_map.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        return [{"doc_id": doc_id, "rrf_score": score, "rank": idx + 1} for idx, (doc_id, score) in enumerate(ranked[:top_k])]

    def hybrid_search(
        self,
        query: str,
        dense_results: List[Dict[str, Any]],
        top_k: int = 5,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> List[Dict[str, Any]]:
        sparse_results = self.search(query, top_k=top_k * 2)
        if self.rrf_enabled and len(dense_results) > 0 and len(sparse_results) > 0:
            rrf_results = self._reciprocal_rank_fusion(dense_results, sparse_results, top_k=top_k)
            id_to_dense = {str(d.get("id", "")): d for d in dense_results}
            ranked = []
            for rrf_item in rrf_results:
                doc_id = rrf_item["doc_id"]
                if doc_id in id_to_dense:
                    doc = id_to_dense[doc_id]
                    result = {**doc, "score": rrf_item["rrf_score"], "fusion_sources": ["dense", "sparse_rrf"]}
                else:
                    result = {"id": doc_id, "score": rrf_item["rrf_score"], "source": "sparse_only", "fusion_sources": ["sparse_rrf"]}
                ranked.append(result)
            return ranked
        sparse_score_map = {r["doc_id"]: r["score"] * sparse_weight for r in sparse_results}
        merged = {}
        for doc in dense_results:
            doc_id = str(doc.get("id", ""))
            merged[doc_id] = {
                **doc,
                "score": doc.get("score", 0.0) * dense_weight,
                "fusion_sources": ["dense"],
            }
        for sparse in sparse_results:
            doc_id = sparse["doc_id"]
            score = sparse["score"] * sparse_weight
            if doc_id in merged:
                merged[doc_id]["score"] += score
                merged[doc_id]["fusion_sources"].append("sparse")
            else:
                merged[doc_id] = {
                    "id": doc_id,
                    "score": score,
                    "source": "sparse_only",
                    "fusion_sources": ["sparse"],
                }
        ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

_sparse_retriever_instance = None

def get_sparse_retriever() -> SparseRetriever:
    global _sparse_retriever_instance
    if _sparse_retriever_instance is None:
        _sparse_retriever_instance = SparseRetriever()
    return _sparse_retriever_instance
