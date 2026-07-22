import uuid
import logging
from typing import List, Dict, Any, Optional
from config import settings

logger = logging.getLogger("crossmind.vector_store")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest_models
    QDRANT_CLIENT_INSTALLED = True
except ImportError:
    QDRANT_CLIENT_INSTALLED = False

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
                    quantization_config=rest_models.ScalarQuantization(
                        scalar=rest_models.ScalarQuantizationConfig(
                            type=rest_models.ScalarType.INT8,
                            quantile=0.99,
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
        Inserts or updates vector records in Qdrant.
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
            payload = item.get("payload", {})

            ids.append(point_id)

            if self.client and QDRANT_CLIENT_INSTALLED:
                qdrant_points.append(
                    rest_models.PointStruct(
                        id=point_id,
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
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic search with inline RBAC and domain payload filtering.
        """
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

        results = []

        if self.client and QDRANT_CLIENT_INSTALLED:
            try:
                query_filter = rest_models.Filter(must=must_filters) if must_filters else None
                
                search_res = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=top_k
                )

                for hit in search_res:
                    results.append({
                        "id": str(hit.id),
                        "score": float(hit.score),
                        "payload": hit.payload
                    })
                return results
            except Exception as e:
                logger.error(f"Error during Qdrant search: {e}. Falling back to memory search.")

        # Fallback memory cosine search with RBAC filter
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
        return scored[:top_k]

_qdrant_engine_instance = None

def get_qdrant_engine() -> QdrantVectorEngine:
    global _qdrant_engine_instance
    if _qdrant_engine_instance is None:
        _qdrant_engine_instance = QdrantVectorEngine()
    return _qdrant_engine_instance
