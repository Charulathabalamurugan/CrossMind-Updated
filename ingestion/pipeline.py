import uuid
import logging
from typing import List, Dict, Any, Optional
from ingestion.embedding import get_embedder
from vector_store.qdrant_engine import get_qdrant_engine

logger = logging.getLogger("crossmind.ingestion")

class IngestionPipeline:
    """
    Phase 1: Unified Multimodal Ingestion Pipeline.
    Processes documents, computes Matryoshka 256-dim embeddings,
    and writes to Qdrant vector database.
    """
    def __init__(self):
        self.embedder = get_embedder()
        self.vector_engine = get_qdrant_engine()

    def ingest_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Accepts a list of document dicts:
        [
            {
                "title": str,
                "content": str,
                "domain": str, (e.g. "neuroscience", "nanotechnology", "pharmacology")
                "year": int,
                "authors": List[str],
                "allowed_roles": List[str], (e.g. ["researcher", "admin"])
                "tags": List[str]
            }
        ]
        """
        if not documents:
            return []

        texts = []
        for doc in documents:
            text_to_embed = f"{doc.get('title', '')} {doc.get('content', '')} {doc.get('domain', '')} {' '.join(doc.get('tags', []))}"
            texts.append(text_to_embed)

        logger.info(f"Generating 256-dim embeddings for {len(texts)} document chunks...")
        embeddings = self.embedder.embed_texts(texts)

        records_to_upsert = []
        for doc, emb in zip(documents, embeddings):
            doc_id = doc.get("id") or str(uuid.uuid4())
            payload = {
                "id": doc_id,
                "title": doc.get("title", "Untitled Document"),
                "content": doc.get("content", ""),
                "domain": doc.get("domain", "general"),
                "year": doc.get("year", 2024),
                "authors": doc.get("authors", []),
                "allowed_roles": doc.get("allowed_roles", ["public", "researcher"]),
                "tags": doc.get("tags", []),
                "citation": f"{doc.get('authors', ['CrossMind Research'])[0]} et al. ({doc.get('year', 2024)}) - {doc.get('title', 'Untitled')}"
            }
            records_to_upsert.append({
                "id": doc_id,
                "vector": emb,
                "payload": payload
            })

        inserted_ids = self.vector_engine.upsert_vectors(records_to_upsert)
        logger.info(f"Successfully ingested {len(inserted_ids)} document records into Qdrant vector store.")
        return inserted_ids

_pipeline_instance = None

def get_ingestion_pipeline() -> IngestionPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = IngestionPipeline()
    return _pipeline_instance
