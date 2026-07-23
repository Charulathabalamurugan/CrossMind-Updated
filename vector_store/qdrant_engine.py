import uuid
import logging
import math
import collections
from typing import List, Dict, Any, Optional
from config import settings

logger = logging.getLogger("crossmind.vector_store")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest_models
    QDRANT_CLIENT_INSTALLED = True
except ImportError:
    QDRANT_CLIENT_INSTALLED = False

class BM25Engine:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.doc_lengths = {}
        self.avg_doc_length = 0.0
        self.doc_term_freqs = {} # doc_id -> term -> count
        self.doc_payloads = {} # doc_id -> payload/record
        self.doc_freqs = collections.defaultdict(int) # term -> doc count
        
    def _tokenize(self, text: str) -> list:
        import re
        if not text:
            return []
        return re.findall(r'\w+', text.lower())

    def index_document(self, doc_id: str, text: str, payload: dict):
        tokens = self._tokenize(text)
        if not tokens:
            return
        
        if doc_id in self.doc_lengths:
            self.doc_count -= 1
            for term, count in self.doc_term_freqs[doc_id].items():
                self.doc_freqs[term] -= 1
                if self.doc_freqs[term] <= 0:
                    del self.doc_freqs[term]

        self.doc_lengths[doc_id] = len(tokens)
        self.doc_payloads[doc_id] = payload
        self.doc_count += 1
        
        tf = collections.defaultdict(int)
        for token in tokens:
            tf[token] += 1
            
        self.doc_term_freqs[doc_id] = tf
        for token in tf.keys():
            self.doc_freqs[token] += 1
            
        total_len = sum(self.doc_lengths.values())
        self.avg_doc_length = total_len / self.doc_count if self.doc_count > 0 else 0.0

    def search(self, query: str, allowed_domains: list = None, user_role: str = "researcher") -> list:
        query_tokens = self._tokenize(query)
        if not query_tokens or self.doc_count == 0:
            return []
            
        scores = []
        for doc_id, tf in self.doc_term_freqs.items():
            payload = self.doc_payloads[doc_id]
            domain = payload.get("domain", "")
            
            if allowed_domains and domain and domain not in allowed_domains:
                continue
                
            if user_role != "admin":
                roles = payload.get("allowed_roles", ["public", "researcher"])
                if not any(r in roles for r in [user_role, "public", "researcher"]):
                    continue
            
            doc_len = self.doc_lengths[doc_id]
            score = 0.0
            for token in query_tokens:
                if token in tf:
                    n = self.doc_freqs[token]
                    idf = math.log(1.0 + (self.doc_count - n + 0.5) / (n + 0.5))
                    f = tf[token]
                    numerator = f * (self.k1 + 1.0)
                    denominator = f + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_length))
                    score += idf * (numerator / denominator)
            if score > 0.0:
                scores.append({
                    "id": doc_id,
                    "score": score,
                    "payload": payload
                })
        if scores:
            max_score = max(x["score"] for x in scores)
            if max_score > 0:
                for s in scores:
                    s["score"] = s["score"] / max_score
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores

class QdrantVectorEngine:
    """
    Phase 2: Secure Vector Retrieval Engine powered by Qdrant.
    Supports HNSW indexing, 256-dim Matryoshka vectors, scalar quantization,
    and inline RBAC payload filtering.
    """
    def __init__(self):
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.dim = settings.EMBEDDING_DIM
        self.client = None
        self._memory_store = [] # Fallback in-memory store if qdrant client isn't running
        self.bm25 = BM25Engine()
        self._init_qdrant()

    def _init_qdrant(self):
        if not QDRANT_CLIENT_INSTALLED:
            logger.warning("qdrant-client not available. Operating in local memory fallback mode.")
            return

        try:
            if settings.QDRANT_IN_MEMORY:
                logger.info("Initializing Qdrant Edge in-memory instance.")
                self.client = QdrantClient(":memory:")
            else:
                logger.info(f"Connecting to Qdrant server at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
                self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

            # Ensure collection exists
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                logger.info(f"Creating Qdrant collection '{self.collection_name}' with vector size {self.dim} & HNSW index")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest_models.VectorParams(
                        size=self.dim,
                        distance=rest_models.Distance.COSINE
                    ),
                    hnsw_config=rest_models.HnswConfigDiff(
                        m=16,
                        ef_construct=100
                    ),
                    optimizers_config=rest_models.OptimizersConfigDiff(
                        default_segment_number=2
                    ),
                    quantization_config=rest_models.ProductQuantization(
                        product=rest_models.ProductQuantizationConfig(
                            compression=rest_models.CompressionRatio.X4,
                            always_ram=True
                        )
                    )
                )
                logger.info("Collection created successfully with scalar quantization.")
        except Exception as e:
            logger.warning(f"Failed to initialize Qdrant client ({e}). Reverting to memory storage.")
            self.client = None

    def upsert_vectors(self, records: List[Dict[str, Any]]) -> List[str]:
        """
        Inserts or updates vector records in Qdrant and indices them in BM25.
        Each record should contain:
          - id (optional, generated if missing)
          - vector (List[float] of dim 256)
          - payload (Dict containing text, domain, roles, metadata)
        """
        ids = []
        qdrant_points = []

        for item in records:
            point_id = item.get("id") or str(uuid.uuid4())
            vector = item["vector"]
            payload = dict(item.get("payload", {}))
            payload.setdefault("id", point_id)

            ids.append(point_id)

            # Index document in BM25
            title = payload.get("title", "")
            content = payload.get("content", "")
            tags = " ".join(payload.get("tags", []))
            self.bm25.index_document(point_id, f"{title} {content} {tags}", payload)

            if self.client and QDRANT_CLIENT_INSTALLED:
                # Qdrant point IDs must be UUIDs or integers. Preserve the
                # external, human-readable document ID in the payload.
                qdrant_point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(point_id)))
                qdrant_points.append(
                    rest_models.PointStruct(
                        id=qdrant_point_id,
                        vector=vector,
                        payload=payload
                    )
                )

            # Store in local fallback store as well
            self._memory_store.append({
                "id": point_id,
                "vector": vector,
                "payload": payload
            })

        if self.client and qdrant_points:
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=qdrant_points
                )
                logger.info(f"Upserted {len(qdrant_points)} vectors into Qdrant collection '{self.collection_name}'")
            except Exception as e:
                logger.error(f"Failed Qdrant upsert: {e}")

        return ids

    def search_with_rbac(
        self,
        query_vector: List[float],
        user_role: str = "researcher",
        allowed_domains: Optional[List[str]] = None,
        top_k: int = 5,
        query_text: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic search with inline RBAC and domain payload filtering.
        If multiple domains are allowed, processes them in parallel across domains.
        """
        if allowed_domains and len(allowed_domains) > 1:
            from concurrent.futures import ThreadPoolExecutor
            
            def search_single_domain(domain):
                return self.search_with_rbac(
                    query_vector=query_vector,
                    user_role=user_role,
                    allowed_domains=[domain],
                    top_k=top_k,
                    query_text=query_text
                )
                
            with ThreadPoolExecutor() as executor:
                domain_results = list(executor.map(search_single_domain, allowed_domains))
                
            # Merge, deduplicate, and sort
            merged = {}
            for res_list in domain_results:
                for item in res_list:
                    doc_id = item["id"]
                    if doc_id not in merged or item["score"] > merged[doc_id]["score"]:
                        merged[doc_id] = item
            
            final_results = list(merged.values())
            final_results.sort(key=lambda x: x["score"], reverse=True)
            return final_results[:top_k]

        # Build filter conditions
        must_filters = []

        # Role-based access control filter
        # Document is accessible if user_role in payload.allowed_roles or 'public' in allowed_roles
        if user_role != "admin":
            role_filter = rest_models.FieldCondition(
                key="allowed_roles",
                match=rest_models.MatchAny(any=[user_role, "public", "researcher"])
            ) if QDRANT_CLIENT_INSTALLED else None
            if role_filter:
                must_filters.append(role_filter)

        # Domain filter if specified
        if allowed_domains and QDRANT_CLIENT_INSTALLED:
            domain_filter = rest_models.FieldCondition(
                key="domain",
                match=rest_models.MatchAny(any=allowed_domains)
            )
            must_filters.append(domain_filter)

        dense_results = []

        if self.client and QDRANT_CLIENT_INSTALLED:
            try:
                query_filter = rest_models.Filter(must=must_filters) if must_filters else None
                
                if hasattr(self.client, "search"):
                    search_res = self.client.search(
                        collection_name=self.collection_name,
                        query_vector=query_vector,
                        query_filter=query_filter,
                        limit=top_k * 2
                    )
                else:
                    search_res = self.client.query_points(
                        collection_name=self.collection_name,
                        query=query_vector,
                        query_filter=query_filter,
                        limit=top_k * 2
                    ).points

                for hit in search_res:
                    dense_results.append({
                        "id": str(hit.payload.get("id", hit.id)),
                        "score": float(hit.score),
                        "payload": hit.payload
                    })
            except Exception as e:
                logger.error(f"Error during Qdrant search: {e}. Falling back to memory search.")

        # Fallback memory cosine search with RBAC filter
        if not dense_results:
            import numpy as np
            q_vec = np.array(query_vector, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec = q_vec / q_norm

            scored = []
            for item in self._memory_store:
                payload = item["payload"]
                roles = payload.get("allowed_roles", ["public", "researcher"])
                domain = payload.get("domain", "")

                # Role check
                if user_role != "admin" and not any(r in roles for r in [user_role, "public", "researcher"]):
                    continue

                # Domain check
                if allowed_domains and domain and domain not in allowed_domains:
                    continue

                v = np.array(item["vector"], dtype=np.float32)
                v_norm = np.linalg.norm(v)
                if v_norm > 0:
                    v = v / v_norm
                score = float(np.dot(q_vec, v))
                scored.append({
                    "id": item["id"],
                    "score": score,
                    "payload": payload
                })

            scored.sort(key=lambda x: x["score"], reverse=True)
            dense_results = scored[:top_k * 2]

        if query_text:
            bm25_results = self.bm25.search(query_text, allowed_domains, user_role)
            all_docs = {}
            for doc in dense_results:
                all_docs[doc["id"]] = doc
            for doc in bm25_results:
                all_docs[doc["id"]] = doc
                
            rrf_scores = {}
            for doc_id in all_docs:
                dense_rank = 9999
                for idx, doc in enumerate(dense_results):
                    if doc["id"] == doc_id:
                        dense_rank = idx
                        break
                bm25_rank = 9999
                for idx, doc in enumerate(bm25_results):
                    if doc["id"] == doc_id:
                        bm25_rank = idx
                        break
                rrf_scores[doc_id] = (1.0 / (60.0 + dense_rank)) + (1.0 / (60.0 + bm25_rank))
            
            merged = []
            for doc_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
                doc_item = dict(all_docs[doc_id])
                doc_item["score"] = rrf_score * 30.0
                merged.append(doc_item)
            return merged[:top_k]
        else:
            return dense_results[:top_k]

_qdrant_engine_instance = None

def get_qdrant_engine() -> QdrantVectorEngine:
    global _qdrant_engine_instance
    if _qdrant_engine_instance is None:
        _qdrant_engine_instance = QdrantVectorEngine()
    return _qdrant_engine_instance
