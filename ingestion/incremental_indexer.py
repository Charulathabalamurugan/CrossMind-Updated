"""Incremental Knowledge Indexer with delta processing, conflict detection, and audit trail."""

import hashlib
import json
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("crossmind.incremental_indexer")


class OperationType(Enum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"


class ConflictStrategy(Enum):
    ABORT = "abort"
    OVERWRITE = "overwrite"
    MERGE = "merge"
    KEEP_EXISTING = "keep_existing"


@dataclass
class DocumentDelta:
    doc_id: str
    operation: OperationType
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_hash: Optional[str] = None
    version: int = 1
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"

    def __post_init__(self):
        if self.content is not None and self.content_hash is None:
            self.content_hash = self._compute_hash(self.content)
        if self.content_hash is None:
            self.content_hash = ""

    @staticmethod
    def _compute_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


@dataclass
class IndexedDocument:
    doc_id: str
    content: str
    content_hash: str
    metadata: Dict[str, Any]
    version: int
    indexed_at: float
    updated_at: float
    is_tombstone: bool = False
    deleted_at: Optional[float] = None


@dataclass
class AuditEvent:
    event_id: str
    timestamp: float
    operation: OperationType
    doc_id: str
    content_hash_before: Optional[str]
    content_hash_after: Optional[str]
    version_before: int
    version_after: int
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "operation": self.operation.value,
            "doc_id": self.doc_id,
            "content_hash_before": self.content_hash_before,
            "content_hash_after": self.content_hash_after,
            "version_before": self.version_before,
            "version_after": self.version_after,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class Checkpoint:
    checkpoint_id: str
    timestamp: float
    document_count: int
    state_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InvalidationEvent:
    event_id: str
    timestamp: float
    doc_ids: Set[str]
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStoreAdapter(ABC):
    @abstractmethod
    def upsert(self, doc_id: str, vector: List[float], payload: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def delete(self, doc_id: str) -> bool:
        pass

    @abstractmethod
    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def search(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def size(self) -> int:
        pass


class GraphStoreAdapter(ABC):
    @abstractmethod
    def add_document(self, doc_id: str, title: str, content: str, domain: str, entities: List[str]) -> None:
        pass

    @abstractmethod
    def remove_document(self, doc_id: str) -> None:
        pass

    @abstractmethod
    def get_entities(self, doc_id: str) -> List[str]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class InMemoryVectorAdapter(VectorStoreAdapter):
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def upsert(self, doc_id: str, vector: List[float], payload: Dict[str, Any]) -> bool:
        with self._lock:
            self._store[doc_id] = {"vector": vector, "payload": payload}
            return True

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id in self._store:
                del self._store[doc_id]
                return True
            return False

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._store.get(doc_id)

    def search(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        with self._lock:
            import numpy as np
            if not self._store:
                return []
            q_vec = np.array(query_vector, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec = q_vec / q_norm

            results = []
            for doc_id, data in self._store.items():
                v = np.array(data["vector"], dtype=np.float32)
                v_norm = np.linalg.norm(v)
                if v_norm > 0:
                    v = v / v_norm
                score = float(np.dot(q_vec, v))
                results.append({"id": doc_id, "score": score, "payload": data["payload"]})
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._store)


class InMemoryGraphAdapter(GraphStoreAdapter):
    def __init__(self):
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._entity_index: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.Lock()

    def add_document(self, doc_id: str, title: str, content: str, domain: str, entities: List[str]) -> None:
        with self._lock:
            self._documents[doc_id] = {
                "title": title,
                "content": content,
                "domain": domain,
                "entities": entities,
            }
            for entity in entities:
                self._entity_index[entity.lower()].add(doc_id)

    def remove_document(self, doc_id: str) -> None:
        with self._lock:
            if doc_id in self._documents:
                for entity in self._documents[doc_id]["entities"]:
                    self._entity_index[entity.lower()].discard(doc_id)
                del self._documents[doc_id]

    def get_entities(self, doc_id: str) -> List[str]:
        with self._lock:
            return self._documents.get(doc_id, {}).get("entities", [])

    def clear(self) -> None:
        with self._lock:
            self._documents.clear()
            self._entity_index.clear()


class IncrementalKnowledgeIndexer:
    def __init__(
        self,
        vector_adapter: Optional[VectorStoreAdapter] = None,
        graph_adapter: Optional[GraphStoreAdapter] = None,
        embedder: Optional[Callable[[str], List[float]]] = None,
        conflict_strategy: ConflictStrategy = ConflictStrategy.ABORT,
        enable_audit: bool = True,
        max_audit_events: int = 10000,
    ):
        self._vector_adapter = vector_adapter or InMemoryVectorAdapter()
        self._graph_adapter = graph_adapter or InMemoryGraphAdapter()
        self._embedder = embedder or (lambda text: [0.0] * 256)
        self._conflict_strategy = conflict_strategy
        self._enable_audit = enable_audit
        self._max_audit_events = max_audit_events

        self._documents: Dict[str, IndexedDocument] = {}
        self._version_map: Dict[str, int] = {}
        self._content_hash_map: Dict[str, str] = {}
        self._tombstones: Set[str] = set()
        self._audit_log: List[AuditEvent] = []
        self._checkpoints: List[Checkpoint] = []
        self._invalidation_events: List[InvalidationEvent] = []
        self._pending_deltas: List[DocumentDelta] = []
        self._lock = threading.RLock()
        self._initialized = False

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            self._initialized = True
            logger.info("IncrementalKnowledgeIndexer initialized")

    def _compute_content_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def _extract_entities(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        entities = list(metadata.get("tags", []))
        title = metadata.get("title", "")
        if title:
            entities.append(title)
        return entities

    def _create_audit_event(
        self,
        operation: OperationType,
        doc_id: str,
        content_hash_before: Optional[str],
        content_hash_after: Optional[str],
        version_before: int,
        version_after: int,
        success: bool,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            operation=operation,
            doc_id=doc_id,
            content_hash_before=content_hash_before,
            content_hash_after=content_hash_after,
            version_before=version_before,
            version_after=version_after,
            success=success,
            error_message=error_message,
            metadata=metadata or {},
        )
        if self._enable_audit:
            self._audit_log.append(event)
            if len(self._audit_log) > self._max_audit_events:
                self._audit_log = self._audit_log[-self._max_audit_events:]
        return event

    def _emit_invalidation(self, doc_ids: Set[str], reason: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        event = InvalidationEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            doc_ids=doc_ids,
            reason=reason,
            metadata=metadata or {},
        )
        self._invalidation_events.append(event)
        logger.info(f"Invalidation event: {reason} for {len(doc_ids)} documents")

    def _detect_conflict(self, delta: DocumentDelta) -> Optional[Dict[str, Any]]:
        existing = self._documents.get(delta.doc_id)
        if not existing:
            return None

        if delta.operation == OperationType.ADD:
            return {
                "type": "duplicate_add",
                "existing_version": existing.version,
                "existing_hash": existing.content_hash,
                "incoming_hash": delta.content_hash,
            }

        if delta.operation == OperationType.UPDATE:
            if existing.version >= delta.version:
                return {
                    "type": "version_conflict",
                    "existing_version": existing.version,
                    "incoming_version": delta.version,
                    "existing_hash": existing.content_hash,
                    "incoming_hash": delta.content_hash,
                }
            if existing.content_hash == delta.content_hash and existing.version == delta.version - 1:
                return None

        if delta.operation == OperationType.DELETE:
            if existing.is_tombstone:
                return {
                    "type": "already_deleted",
                    "existing_version": existing.version,
                    "deleted_at": existing.deleted_at,
                }

        return None

    def _resolve_conflict(self, delta: DocumentDelta, conflict: Dict[str, Any]) -> bool:
        if self._conflict_strategy == ConflictStrategy.ABORT:
            return False
        elif self._conflict_strategy == ConflictStrategy.OVERWRITE:
            return True
        elif self._conflict_strategy == ConflictStrategy.KEEP_EXISTING:
            return False
        elif self._conflict_strategy == ConflictStrategy.MERGE:
            return True
        return False

    def apply_delta(self, delta: DocumentDelta) -> bool:
        with self._lock:
            if not self._initialized:
                self.initialize()

            conflict = self._detect_conflict(delta)
            if conflict:
                resolved = self._resolve_conflict(delta, conflict)
                if not resolved:
                    self._create_audit_event(
                        operation=delta.operation,
                        doc_id=delta.doc_id,
                        content_hash_before=self._documents.get(delta.doc_id, IndexedDocument(
                            doc_id=delta.doc_id, content="", content_hash="", metadata={}, version=0, indexed_at=0, updated_at=0
                        )).content_hash,
                        content_hash_after=delta.content_hash,
                        version_before=self._documents.get(delta.doc_id, IndexedDocument(
                            doc_id=delta.doc_id, content="", content_hash="", metadata={}, version=0, indexed_at=0, updated_at=0
                        )).version,
                        version_after=delta.version,
                        success=False,
                        error_message=f"Conflict detected: {conflict['type']}",
                        metadata={"conflict": conflict, "strategy": self._conflict_strategy.value},
                    )
                    return False

            try:
                if delta.operation == OperationType.ADD:
                    return self._apply_add(delta)
                elif delta.operation == OperationType.UPDATE:
                    return self._apply_update(delta)
                elif delta.operation == OperationType.DELETE:
                    return self._apply_delete(delta)
            except Exception as e:
                logger.error(f"Failed to apply delta for {delta.doc_id}: {e}")
                self._create_audit_event(
                    operation=delta.operation,
                    doc_id=delta.doc_id,
                    content_hash_before=self._documents.get(delta.doc_id, IndexedDocument(
                        doc_id=delta.doc_id, content="", content_hash="", metadata={}, version=0, indexed_at=0, updated_at=0
                    )).content_hash,
                    content_hash_after=delta.content_hash,
                    version_before=self._documents.get(delta.doc_id, IndexedDocument(
                        doc_id=delta.doc_id, content="", content_hash="", metadata={}, version=0, indexed_at=0, updated_at=0
                    )).version,
                    version_after=delta.version,
                    success=False,
                    error_message=str(e),
                )
                return False

    def _apply_add(self, delta: DocumentDelta) -> bool:
        if delta.content is None:
            raise ValueError("Content required for ADD operation")

        content_hash = delta.content_hash or self._compute_content_hash(delta.content)
        now = time.time()
        vector = self._embedder(delta.content)
        entities = self._extract_entities(delta.content, delta.metadata)

        doc = IndexedDocument(
            doc_id=delta.doc_id,
            content=delta.content,
            content_hash=content_hash,
            metadata=delta.metadata,
            version=delta.version,
            indexed_at=now,
            updated_at=now,
        )

        self._documents[delta.doc_id] = doc
        self._version_map[delta.doc_id] = delta.version
        self._content_hash_map[delta.doc_id] = content_hash

        self._vector_adapter.upsert(delta.doc_id, vector, {
            "id": delta.doc_id,
            "content": delta.content,
            "content_hash": content_hash,
            "version": delta.version,
            "metadata": delta.metadata,
        })

        self._graph_adapter.add_document(
            delta.doc_id,
            delta.metadata.get("title", delta.doc_id),
            delta.content,
            delta.metadata.get("domain", "general"),
            entities,
        )

        self._create_audit_event(
            operation=OperationType.ADD,
            doc_id=delta.doc_id,
            content_hash_before=None,
            content_hash_after=content_hash,
            version_before=0,
            version_after=delta.version,
            success=True,
            metadata={"source": delta.source},
        )
        return True

    def _apply_update(self, delta: DocumentDelta) -> bool:
        existing = self._documents.get(delta.doc_id)
        if not existing:
            raise ValueError(f"Document {delta.doc_id} not found for UPDATE")

        if delta.content is None:
            raise ValueError("Content required for UPDATE operation")

        content_hash = delta.content_hash or self._compute_content_hash(delta.content)
        now = time.time()
        vector = self._embedder(delta.content)
        entities = self._extract_entities(delta.content, delta.metadata)

        old_hash = existing.content_hash
        old_version = existing.version

        existing.content = delta.content
        existing.content_hash = content_hash
        existing.metadata = delta.metadata
        existing.version = delta.version
        existing.updated_at = now

        self._version_map[delta.doc_id] = delta.version
        self._content_hash_map[delta.doc_id] = content_hash

        self._vector_adapter.upsert(delta.doc_id, vector, {
            "id": delta.doc_id,
            "content": delta.content,
            "content_hash": content_hash,
            "version": delta.version,
            "metadata": delta.metadata,
        })

        self._graph_adapter.remove_document(delta.doc_id)
        self._graph_adapter.add_document(
            delta.doc_id,
            delta.metadata.get("title", delta.doc_id),
            delta.content,
            delta.metadata.get("domain", "general"),
            entities,
        )

        self._create_audit_event(
            operation=OperationType.UPDATE,
            doc_id=delta.doc_id,
            content_hash_before=old_hash,
            content_hash_after=content_hash,
            version_before=old_version,
            version_after=delta.version,
            success=True,
            metadata={"source": delta.source},
        )
        return True

    def _apply_delete(self, delta: DocumentDelta) -> bool:
        existing = self._documents.get(delta.doc_id)
        if not existing:
            self._create_audit_event(
                operation=OperationType.DELETE,
                doc_id=delta.doc_id,
                content_hash_before=None,
                content_hash_after=None,
                version_before=0,
                version_after=0,
                success=False,
                error_message="Document not found",
            )
            return False

        old_hash = existing.content_hash
        old_version = existing.version

        existing.is_tombstone = True
        existing.deleted_at = time.time()
        existing.version = delta.version
        existing.updated_at = existing.deleted_at

        self._tombstones.add(delta.doc_id)
        self._version_map[delta.doc_id] = delta.version

        self._vector_adapter.delete(delta.doc_id)
        self._graph_adapter.remove_document(delta.doc_id)

        self._create_audit_event(
            operation=OperationType.DELETE,
            doc_id=delta.doc_id,
            content_hash_before=old_hash,
            content_hash_after=None,
            version_before=old_version,
            version_after=delta.version,
            success=True,
            metadata={"source": delta.source},
        )
        return True

    def apply_deltas(self, deltas: List[DocumentDelta]) -> Dict[str, Any]:
        results = {"success": [], "failed": [], "conflicts": []}
        for delta in deltas:
            success = self.apply_delta(delta)
            if success:
                results["success"].append(delta.doc_id)
            else:
                results["failed"].append(delta.doc_id)
                conflict = self._detect_conflict(delta)
                if conflict:
                    results["conflicts"].append({"doc_id": delta.doc_id, "conflict": conflict})
        return results

    def get_document(self, doc_id: str) -> Optional[IndexedDocument]:
        with self._lock:
            doc = self._documents.get(doc_id)
            if doc and doc.is_tombstone:
                return None
            return doc

    def get_document_with_tombstone(self, doc_id: str) -> Optional[IndexedDocument]:
        with self._lock:
            return self._documents.get(doc_id)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            query_vector = self._embedder(query)
            return self._vector_adapter.search(query_vector, top_k)

    def create_checkpoint(self, metadata: Optional[Dict[str, Any]] = None) -> Checkpoint:
        with self._lock:
            state_data = {
                "documents": {k: {"content": v.content, "content_hash": v.content_hash, "metadata": v.metadata, "version": v.version, "indexed_at": v.indexed_at, "updated_at": v.updated_at} for k, v in self._documents.items() if not v.is_tombstone},
                "versions": dict(self._version_map),
                "tombstones": list(self._tombstones),
            }
            state_json = json.dumps(state_data, sort_keys=True, default=str)
            state_hash = hashlib.sha256(state_json.encode("utf-8")).hexdigest()[:16]

            checkpoint = Checkpoint(
                checkpoint_id=str(uuid.uuid4()),
                timestamp=time.time(),
                document_count=len([d for d in self._documents.values() if not d.is_tombstone]),
                state_hash=state_hash,
                metadata=dict(metadata or {}, **{"_state": state_data}),
            )
            self._checkpoints.append(checkpoint)
            logger.info(f"Created checkpoint {checkpoint.checkpoint_id} with {checkpoint.document_count} documents")
            return checkpoint

    def rollback_to_checkpoint(self, checkpoint_id: str) -> bool:
        with self._lock:
            checkpoint = next((c for c in self._checkpoints if c.checkpoint_id == checkpoint_id), None)
            if not checkpoint:
                logger.error(f"Checkpoint {checkpoint_id} not found")
                return False

            state_data = checkpoint.metadata.get("_state", {})
            documents_state = state_data.get("documents", {})
            versions_state = state_data.get("versions", {})
            tombstones_state = state_data.get("tombstones", [])

            self._documents.clear()
            self._version_map.clear()
            self._content_hash_map.clear()
            self._tombstones.clear()
            self._vector_adapter.clear()
            self._graph_adapter.clear()

            for doc_id, doc_data in documents_state.items():
                doc = IndexedDocument(
                    doc_id=doc_id,
                    content=doc_data["content"],
                    content_hash=doc_data["content_hash"],
                    metadata=doc_data["metadata"],
                    version=doc_data["version"],
                    indexed_at=doc_data["indexed_at"],
                    updated_at=doc_data["updated_at"],
                    is_tombstone=False,
                )
                self._documents[doc_id] = doc
                self._version_map[doc_id] = doc_data["version"]
                self._content_hash_map[doc_id] = doc_data["content_hash"]

                vector = self._embedder(doc_data["content"])
                entities = self._extract_entities(doc_data["content"], doc_data["metadata"])

                self._vector_adapter.upsert(doc_id, vector, {
                    "id": doc_id,
                    "content": doc_data["content"],
                    "content_hash": doc_data["content_hash"],
                    "version": doc_data["version"],
                    "metadata": doc_data["metadata"],
                })

                self._graph_adapter.add_document(
                    doc_id,
                    doc_data["metadata"].get("title", doc_id),
                    doc_data["content"],
                    doc_data["metadata"].get("domain", "general"),
                    entities,
                )

            for doc_id in tombstones_state:
                self._tombstones.add(doc_id)

            logger.info(f"Rolled back to checkpoint {checkpoint_id} with {len(documents_state)} documents")
            return True

    def get_audit_log(self, limit: int = 100, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            events = self._audit_log
            if doc_id:
                events = [e for e in events if e.doc_id == doc_id]
            return [e.to_dict() for e in events[-limit:]]

    def get_invalidation_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "event_id": e.event_id,
                    "timestamp": e.timestamp,
                    "doc_ids": list(e.doc_ids),
                    "reason": e.reason,
                    "metadata": e.metadata,
                }
                for e in self._invalidation_events[-limit:]
            ]

    def get_checkpoints(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "checkpoint_id": c.checkpoint_id,
                    "timestamp": c.timestamp,
                    "document_count": c.document_count,
                    "state_hash": c.state_hash,
                    "metadata": c.metadata,
                }
                for c in self._checkpoints
            ]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            active_docs = [d for d in self._documents.values() if not d.is_tombstone]
            return {
                "total_documents": len(self._documents),
                "active_documents": len(active_docs),
                "tombstones": len(self._tombstones),
                "audit_events": len(self._audit_log),
                "checkpoints": len(self._checkpoints),
                "invalidation_events": len(self._invalidation_events),
                "vector_store_size": self._vector_adapter.size(),
            }

    def invalidate_documents(self, doc_ids: Set[str], reason: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            self._emit_invalidation(doc_ids, reason, metadata)
            for doc_id in doc_ids:
                if doc_id in self._documents:
                    self._documents[doc_id].is_tombstone = True
                    self._documents[doc_id].deleted_at = time.time()
                    self._tombstones.add(doc_id)
                    self._vector_adapter.delete(doc_id)
                    self._graph_adapter.remove_document(doc_id)

    def clear(self) -> None:
        with self._lock:
            self._documents.clear()
            self._version_map.clear()
            self._content_hash_map.clear()
            self._tombstones.clear()
            self._audit_log.clear()
            self._checkpoints.clear()
            self._invalidation_events.clear()
            self._vector_adapter.clear()
            self._graph_adapter.clear()


_indexer_instance: Optional[IncrementalKnowledgeIndexer] = None


def get_incremental_indexer(
    vector_adapter: Optional[VectorStoreAdapter] = None,
    graph_adapter: Optional[GraphStoreAdapter] = None,
    embedder: Optional[Callable[[str], List[float]]] = None,
    conflict_strategy: ConflictStrategy = ConflictStrategy.ABORT,
) -> IncrementalKnowledgeIndexer:
    global _indexer_instance
    if _indexer_instance is None:
        _indexer_instance = IncrementalKnowledgeIndexer(
            vector_adapter=vector_adapter,
            graph_adapter=graph_adapter,
            embedder=embedder,
            conflict_strategy=conflict_strategy,
        )
    return _indexer_instance