import time
import logging
from typing import List, Dict, Any, Optional
from ingestion.sparse_vector import get_sparse_vector_engine

logger = logging.getLogger("crossmind.sparse_retriever")

class SparseRetriever:
    def __init__(self):
        self.sparse_engine = get_sparse_vector_engine()

    def index_documents(self, documents: List[Dict[str, Any]]):
        self.sparse_engine.index_documents(documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.sparse_engine.search(query, top_k=top_k)
        for r in results:
            r["source"] = "sparse_tfidf"
        return results

    def hybrid_search(
        self,
        query: str,
        dense_results: List[Dict[str, Any]],
        top_k: int = 5,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> List[Dict[str, Any]]:
        sparse_results = self.search(query, top_k=top_k * 2)
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
