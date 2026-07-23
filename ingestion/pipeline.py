import uuid
import logging
from typing import List, Dict, Any, Optional
from config import settings
from ingestion.embedding import get_embedder
from vector_store.qdrant_engine import get_qdrant_engine
from reasoning.knowledge_graph import get_knowledge_graph
from ingestion.active_learning import get_active_learning_engine
from ingestion.ingestion_cache import get_ingestion_cache

logger = logging.getLogger("crossmind.ingestion")

class IngestionPipeline:
    """
    Phase 1: Unified Multimodal Ingestion Pipeline.
    Processes documents, computes DSKE embeddings,
    and writes to Qdrant vector database with caching, active learning,
    and continuous ingestion support.
    """
    def __init__(self):
        self.embedder = get_embedder()
        self.vector_engine = get_qdrant_engine()
        self.knowledge_graph = get_knowledge_graph()
        self.cache = get_ingestion_cache()
        self.active_learning = get_active_learning_engine()
        self._initialized = False

    def auto_init(self):
        if self._initialized:
            return
        self._initialized = True
        if not settings.AUTO_INIT_ON_STARTUP:
            logger.info("Auto-init disabled by settings.")
            return
        from ingestion.dynamic_connectors import get_dynamic_connectors
        manager = get_dynamic_connectors()
        manager.load_from_env(callback=self.ingest_documents)
        manager.start_all(callback=self.ingest_documents)
        self._start_continuous_ingestion()
        logger.info("Auto-init complete: connectors and continuous ingestion started.")

    def _start_continuous_ingestion(self):
        try:
            from ingestion.continuous_ingestion import ContinuousIngestionWorker
            worker = ContinuousIngestionWorker(pipeline=self)
            worker.start()
            self._continuous_worker = worker
        except Exception as exc:
            logger.warning(f"Failed to start continuous ingestion worker: {exc}")

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

        deduped: List[Dict[str, Any]] = []
        for doc in documents:
            cache_key = doc.get("content_hash") or str(hash(doc.get("content", "")))
            if self.cache.get(cache_key):
                continue
            if not doc.get("content_hash"):
                doc["content_hash"] = cache_key
            deduped.append(doc)
            self.cache.set(cache_key, True)

        texts = []
        for doc in deduped:
            text_to_embed = f"{doc.get('title', '')} {doc.get('content', '')} {doc.get('domain', '')} {' '.join(doc.get('tags', []))}"
            texts.append(text_to_embed)

        logger.info(f"Generating DSKE embeddings for {len(texts)} document chunks...")
        embeddings = self.embedder.embed_texts(texts)

        records_to_upsert = []
        for doc, emb in zip(deduped, embeddings):
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
                "citation": f"{doc.get('authors', ['CrossMind Research'])[0]} et al. ({doc.get('year', 2024)}) - {doc.get('title', 'Untitled')}",
            }
            records_to_upsert.append({
                "id": doc_id,
                "vector": emb,
                "payload": payload,
            })

        inserted_ids = self.vector_engine.upsert_vectors(records_to_upsert)
        self.knowledge_graph.index_documents([record["payload"] for record in records_to_upsert])
        for record in records_to_upsert:
            cache_key = record["payload"].get("content_hash") or str(hash(record["payload"].get("content", "")))
            self.cache.set(cache_key, record["payload"])
        logger.info(f"Successfully ingested {len(inserted_ids)} document records into Qdrant vector store.")
        return inserted_ids

    def record_feedback(self, query: str, doc_id: str, score: float, user_role: str):
        self.active_learning.record_feedback(query, doc_id, score, user_role)
        self.active_learning.retrain(self)

_pipeline_instance = None


def get_ingestion_pipeline() -> IngestionPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = IngestionPipeline()
    return _pipeline_instance
