import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger("crossmind.celery")

# Optional Celery setup for distributed async ingestion tasks
try:
    from celery import Celery
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    celery_app = Celery("crossmind_tasks", broker=redis_url, backend=redis_url)
    CELERY_AVAILABLE = True
except Exception:
    CELERY_AVAILABLE = False
    celery_app = None

def process_batch_embedding_job(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Task function for processing document batch embeddings and indexing into Qdrant.
    """
    from ingestion.pipeline import get_ingestion_pipeline
    pipeline = get_ingestion_pipeline()
    indexed_ids = pipeline.ingest_documents(documents)
    return {
        "status": "completed",
        "processed_count": len(indexed_ids),
        "document_ids": indexed_ids
    }

if CELERY_AVAILABLE and celery_app:
    @celery_app.task(name="tasks.ingest_documents_task")
    def ingest_documents_task(documents: List[Dict[str, Any]]):
        return process_batch_embedding_job(documents)
