"""Trace where the dense search is failing."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion.pipeline import get_ingestion_pipeline
from vector_store.qdrant_engine import get_qdrant_engine
from ingestion.embedding import get_embedder
from vector_store.vector_adapter import get_vector_adapter
from config import settings

pipeline = get_ingestion_pipeline()
docs = [
    {"id": "doc1", "title": "Lithium-Ion Battery Anode Degradation", "domain": "energy", "year": 2024, "authors": ["Smith"], "content": "Abstract: lithium-ion battery anode degradation. SEI layer growth increases resistance. Silicon anodes improve cycle life.", "tags": ["battery", "lithium"]},
    {"id": "doc2", "title": "Solid-State Battery Electrolytes", "domain": "energy", "year": 2025, "authors": ["Chen"], "content": "Abstract: solid-state electrolytes improve safety. Ionic conductivity 10 mS/cm.", "tags": ["battery", "solid-state"]},
]
pipeline.ingest_documents(docs)

engine = get_qdrant_engine()
embedder = get_embedder()
adapter = get_vector_adapter()

query = "How does silicon improve lithium battery anodes?"
qv = embedder.embed_text(query, dim=settings.BGE_M3_RETRIEVAL_DIM if settings.BGE_M3_MATRYOSHKA_ENABLED else settings.EMBEDDING_DIM)
qn = adapter.normalize(qv, force_dim=settings.BGE_M3_RETRIEVAL_DIM if settings.BGE_M3_MATRYOSHKA_ENABLED else settings.EMBEDDING_DIM)
print("Query vector dim:", len(qn.get("flat_vector", qv)))

# Direct call with empty allowed_domains to bypass domain filter
res = engine.search_with_rbac(
    query_vector=qn.get("flat_vector", qv),
    user_role="admin",
    allowed_domains=None,
    top_k=5,
    query_text=query,
)
print("Direct search results:", len(res))
for r in res:
    print(f"  {r['id']}: {r['payload'].get('title')} score={r['score']:.3f}")
