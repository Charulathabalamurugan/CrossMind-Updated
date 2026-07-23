import time
import logging
from typing import List, Dict, Any, Optional
from reasoning.knowledge_graph import KnowledgeGraph, DiscoveryScorer
from vector_store.qdrant_engine import get_qdrant_engine
from ingestion.embedding import get_embedder

logger = logging.getLogger("crossmind.hybrid_rag_kg")

class HybridRAGKG:
    def __init__(self):
        self.knowledge_graph = KnowledgeGraph()
        self.vector_engine = get_qdrant_engine()
        self.embedder = get_embedder()

    def retrieve(self, query: str, user_role: str, allowed_domains: List[str], top_k: int = 5) -> Dict[str, Any]:
        start = time.time()
        query_vector = self.embedder.embed_text(query)
        dense_results = self.vector_engine.search_with_rbac(
            query_vector=query_vector,
            user_role=user_role,
            allowed_domains=allowed_domains,
            top_k=top_k,
            query_text=query,
        )
        dense_ms = round((time.time() - start) * 1000, 2)
        
        graph_start = time.time()
        graph_context = self.knowledge_graph.graph_rag_context(
            dense_results, allowed_domains
        )
        graph_ms = round((time.time() - graph_start) * 1000, 2)
        
        discovery_score = DiscoveryScorer.score(dense_results, graph_context)
        
        fused = []
        seen_ids = set()
        for item in dense_results:
            doc_id = str(item.get("id"))
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            fused.append({
                "id": doc_id,
                "score": item.get("score", 0.0),
                "payload": item.get("payload", {}),
                "retrieval_source": ["dense_vector", "graph_rag"],
            })
        
        for path in graph_context.get("multi_hop_paths", [])[:top_k]:
            for node_id in path.get("path", []):
                if node_id.startswith("doc:") and node_id not in seen_ids:
                    doc_id = node_id.replace("doc:", "")
                    docs = self.knowledge_graph.documents
                    payload = docs.get(doc_id, {})
                    if payload:
                        fused.append({
                            "id": doc_id,
                            "score": path.get("path_score", 0.0) / 100.0,
                            "payload": payload,
                            "retrieval_source": ["graph_rag"],
                            "path_score": path.get("path_score"),
                        })
                        seen_ids.add(doc_id)
        
        fused.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        
        return {
            "fused_results": fused[:top_k],
            "dense_results_count": len(dense_results),
            "graph_nodes_count": len(graph_context.get("nodes", [])),
            "graph_paths_count": len(graph_context.get("multi_hop_paths", [])),
            "dense_retrieval_ms": dense_ms,
            "graph_expansion_ms": graph_ms,
            "discovery_score": discovery_score,
            "graph_context": graph_context,
            "strategy": "hybrid_rag_kg",
        }

_hybrid_instance: Optional[HybridRAGKG] = None

def get_hybrid_rag_kg() -> HybridRAGKG:
    global _hybrid_instance
    if _hybrid_instance is None:
        _hybrid_instance = HybridRAGKG()
    return _hybrid_instance
