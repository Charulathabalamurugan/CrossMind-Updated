from typing import Any, Dict


def ingest_document_task(document: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "queued", "document_id": document.get("id"), "result": document}


def process_batch_task(batch: list) -> Dict[str, Any]:
    return {"status": "processed", "count": len(batch)}


__all__ = ["ingest_document_task", "process_batch_task"]
