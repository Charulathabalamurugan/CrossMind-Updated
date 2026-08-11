import time
import logging
import hashlib
from typing import List, Dict, Any, Optional
from config import settings
from ingestion.sparse_vector import get_sparse_vector_engine
from vector_store.vector_adapter import get_vector_adapter

logger = logging.getLogger("crossmind.sparse_retriever")

class SparseRetriever:
    """
    SparseRetriever orchestrates the BM25 retrieval process for exact keyword search.
    """
    def __init__(self):
        self.sparse_engine = get_sparse_vector_engine() # BM25 engine
        self.rrf_enabled = getattr(settings, "RRF_ENABLED", True)
        self.rrf_k = getattr(settings, "RRF_K", 60)
        self.cache_ttl = getattr(settings, "REDIS_RETRIEVAL_CACHE_TTL", 1800)
        self.cache_max = getattr(settings, "REDIS_RETRIEVAL_CACHE_MAX", 10000)
        self.adapter = get_vector_adapter()
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
        quality_map = {}
        
        # Build quality score map from dense and sparse results payload
        for doc in dense_results:
            doc_id = str(doc.get("id", ""))
            quality_map[doc_id] = doc.get("payload", {}).get("quality_score", 0.5)
            
        for doc in sparse_results:
            doc_id = str(doc.get("doc_id", doc.get("id", "")))
            if doc_id not in quality_map:
                quality_map[doc_id] = doc.get("payload", {}).get("quality_score", 0.5)

        for rank, doc in enumerate(dense_results, 1):
            doc_id = str(doc.get("id", ""))
            score_map[doc_id] = score_map.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
            
        for rank, doc in enumerate(sparse_results, 1):
            doc_id = str(doc.get("doc_id", doc.get("id", "")))
            score_map[doc_id] = score_map.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
            
        # Apply Quality Weighting to RRF scores
        for doc_id in score_map:
            q_score = quality_map.get(doc_id, 0.5)
            score_map[doc_id] = score_map[doc_id] * (1.0 + q_score)
            
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
            id_to_sparse = {str(s.get("doc_id", s.get("id", ""))): s for s in sparse_results}
            ranked = []
            for rrf_item in rrf_results:
                doc_id = rrf_item["doc_id"]
                q_score = 0.5
                if doc_id in id_to_dense:
                    doc = id_to_dense[doc_id]
                    q_score = doc.get("payload", {}).get("quality_score", 0.5)
                    result = {**doc, "score": rrf_item["rrf_score"], "fusion_sources": ["dense", "sparse_rrf"], "quality_score": q_score}
                elif doc_id in id_to_sparse:
                    doc = id_to_sparse[doc_id]
                    q_score = doc.get("payload", {}).get("quality_score", 0.5)
                    result = {
                        "id": doc_id,
                        "score": rrf_item["rrf_score"],
                        "payload": doc.get("payload", {}),
                        "source": "sparse_only",
                        "fusion_sources": ["sparse_rrf"],
                        "quality_score": q_score
                    }
                else:
                    result = {"id": doc_id, "score": rrf_item["rrf_score"], "source": "sparse_only", "fusion_sources": ["sparse_rrf"], "quality_score": q_score}
                ranked.append(result)
            # Re-sort ranked list to ensure proper ordering after fusion scoring
            ranked.sort(key=lambda x: x["score"], reverse=True)
            return ranked
        
        # Fallback to linear combination weight and also apply quality score weighting
        sparse_score_map = {r.get("doc_id", r.get("id", "")): r.get("score", 0.0) * sparse_weight for r in sparse_results}
        merged = {}
        for doc in dense_results:
            doc_id = str(doc.get("id", ""))
            q_score = doc.get("payload", {}).get("quality_score", 0.5)
            merged[doc_id] = {
                **doc,
                "score": doc.get("score", 0.0) * dense_weight * (1.0 + q_score),
                "fusion_sources": ["dense"],
                "quality_score": q_score,
            }
        for sparse in sparse_results:
            doc_id = sparse.get("doc_id", sparse.get("id", ""))
            score = sparse.get("score", 0.0) * sparse_weight
            q_score = sparse.get("payload", {}).get("quality_score", 0.5)
            boosted_score = score * (1.0 + q_score)
            if doc_id in merged:
                merged[doc_id]["score"] += boosted_score
                merged[doc_id]["fusion_sources"].append("sparse")
            else:
                merged[doc_id] = {
                    "id": doc_id,
                    "score": boosted_score,
                    "payload": sparse.get("payload", {}),
                    "source": "sparse_only",
                    "fusion_sources": ["sparse"],
                    "quality_score": q_score,
                }
        ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

_sparse_retriever_instance = None

def get_sparse_retriever() -> SparseRetriever:
    global _sparse_retriever_instance
    if _sparse_retriever_instance is None:
        _sparse_retriever_instance = SparseRetriever()
    return _sparse_retriever_instance
