"""Tests for IncrementalKnowledgeIndexer."""

import pytest
import threading
import time
from typing import List
from unittest.mock import Mock, patch

from ingestion.incremental_indexer import (
    IncrementalKnowledgeIndexer,
    DocumentDelta,
    OperationType,
    ConflictStrategy,
    VectorStoreAdapter,
    GraphStoreAdapter,
    InMemoryVectorAdapter,
    InMemoryGraphAdapter,
    IndexedDocument,
    AuditEvent,
    Checkpoint,
    InvalidationEvent,
)


class TestIncrementalKnowledgeIndexer:
    """Test suite for IncrementalKnowledgeIndexer."""

    def setup_method(self):
        self.indexer = IncrementalKnowledgeIndexer()
        self.indexer.initialize()

    def teardown_method(self):
        self.indexer.clear()

    def test_initialization(self):
        """Test indexer initialization."""
        assert self.indexer._initialized is True
        stats = self.indexer.get_stats()
        assert stats["total_documents"] == 0
        assert stats["active_documents"] == 0

    def test_add_document(self):
        """Test adding a document."""
        delta = DocumentDelta(
            doc_id="doc1",
            operation=OperationType.ADD,
            content="Test document content",
            metadata={"title": "Test Doc", "domain": "test"},
            version=1,
        )
        result = self.indexer.apply_delta(delta)
        assert result is True

        doc = self.indexer.get_document("doc1")
        assert doc is not None
        assert doc.content == "Test document content"
        assert doc.metadata["title"] == "Test Doc"
        assert doc.version == 1
        assert not doc.is_tombstone

    def test_update_document(self):
        """Test updating a document."""
        add_delta = DocumentDelta(
            doc_id="doc1",
            operation=OperationType.ADD,
            content="Original content",
            metadata={"title": "Test Doc"},
            version=1,
        )
        self.indexer.apply_delta(add_delta)

        update_delta = DocumentDelta(
            doc_id="doc1",
            operation=OperationType.UPDATE,
            content="Updated content",
            metadata={"title": "Test Doc Updated"},
            version=2,
        )
        result = self.indexer.apply_delta(update_delta)
        assert result is True

        doc = self.indexer.get_document("doc1")
        assert doc.content == "Updated content"
        assert doc.version == 2
        assert doc.metadata["title"] == "Test Doc Updated"

    def test_delete_document(self):
        """Test deleting a document."""
        add_delta = DocumentDelta(
            doc_id="doc1",
            operation=OperationType.ADD,
            content="Content to delete",
            metadata={},
            version=1,
        )
        self.indexer.apply_delta(add_delta)

        delete_delta = DocumentDelta(
            doc_id="doc1",
            operation=OperationType.DELETE,
            version=2,
        )
        result = self.indexer.apply_delta(delete_delta)
        assert result is True

        doc = self.indexer.get_document("doc1")
        assert doc is None

        doc_with_tombstone = self.indexer.get_document_with_tombstone("doc1")
        assert doc_with_tombstone is not None
        assert doc_with_tombstone.is_tombstone is True
        assert doc_with_tombstone.deleted_at is not None

    def test_idempotent_upsert(self):
        """Test idempotent upsert - same content hash should not create duplicate."""
        delta1 = DocumentDelta(
            doc_id="doc1",
            operation=OperationType.ADD,
            content="Same content",
            metadata={},
            version=1,
        )
        self.indexer.apply_delta(delta1)

        delta2 = DocumentDelta(
            doc_id="doc1",
            operation=OperationType.ADD,
            content="Same content",
            metadata={},
            version=1,
        )
        result = self.indexer.apply_delta(delta2)
        assert result is False

        stats = self.indexer.get_stats()
        assert stats["active_documents"] == 1

    def test_content_hash_verification(self):
        """Test content hash is computed and verified."""
        delta = DocumentDelta(
            doc_id="doc1",
            operation=OperationType.ADD,
            content="Content for hash",
            metadata={},
            version=1,
        )
        self.indexer.apply_delta(delta)

        doc = self.indexer.get_document("doc1")
        expected_hash = self.indexer._compute_content_hash("Content for hash")
        assert doc.content_hash == expected_hash

    def test_version_check_on_update(self):
        """Test version conflict detection on update."""
        add_delta = DocumentDelta(
            doc_id="doc1",
            operation=OperationType.ADD,
            content="Original",
            metadata={},
            version=1,
        )
        self.indexer.apply_delta(add_delta)

        update_delta_old_version = DocumentDelta(
            doc_id="doc1",
            operation=OperationType.UPDATE,
            content="Update with old version",
            metadata={},
            version=1,
        )
        result = self.indexer.apply_delta(update_delta_old_version)
        assert result is False

    def test_conflict_strategy_abort(self):
        """Test ABORT conflict strategy."""
        indexer = IncrementalKnowledgeIndexer(conflict_strategy=ConflictStrategy.ABORT)
        indexer.initialize()

        add_delta = DocumentDelta(doc_id="doc1", operation=OperationType.ADD, content="Original", metadata={}, version=1)
        indexer.apply_delta(add_delta)

        update_delta = DocumentDelta(doc_id="doc1", operation=OperationType.UPDATE, content="Conflict", metadata={}, version=1)
        result = indexer.apply_delta(update_delta)
        assert result is False

    def test_conflict_strategy_overwrite(self):
        """Test OVERWRITE conflict strategy."""
        indexer = IncrementalKnowledgeIndexer(conflict_strategy=ConflictStrategy.OVERWRITE)
        indexer.initialize()

        add_delta = DocumentDelta(doc_id="doc1", operation=OperationType.ADD, content="Original", metadata={}, version=1)
        indexer.apply_delta(add_delta)

        update_delta = DocumentDelta(doc_id="doc1", operation=OperationType.UPDATE, content="Overwritten", metadata={}, version=1)
        result = indexer.apply_delta(update_delta)
        assert result is True

        doc = indexer.get_document("doc1")
        assert doc.content == "Overwritten"

    def test_conflict_strategy_keep_existing(self):
        """Test KEEP_EXISTING conflict strategy."""
        indexer = IncrementalKnowledgeIndexer(conflict_strategy=ConflictStrategy.KEEP_EXISTING)
        indexer.initialize()

        add_delta = DocumentDelta(doc_id="doc1", operation=OperationType.ADD, content="Original", metadata={}, version=1)
        indexer.apply_delta(add_delta)

        update_delta = DocumentDelta(doc_id="doc1", operation=OperationType.UPDATE, content="Should be ignored", metadata={}, version=1)
        result = indexer.apply_delta(update_delta)
        assert result is False

        doc = indexer.get_document("doc1")
        assert doc.content == "Original"

    def test_tombstone_handling(self):
        """Test tombstone creation and handling."""
        add_delta = DocumentDelta(doc_id="doc1", operation=OperationType.ADD, content="To delete", metadata={}, version=1)
        self.indexer.apply_delta(add_delta)

        delete_delta = DocumentDelta(doc_id="doc1", operation=OperationType.DELETE, version=2)
        self.indexer.apply_delta(delete_delta)

        assert "doc1" in self.indexer._tombstones

        delete_again = DocumentDelta(doc_id="doc1", operation=OperationType.DELETE, version=3)
        result = self.indexer.apply_delta(delete_again)
        assert result is False

    def test_batch_deltas(self):
        """Test applying multiple deltas at once."""
        deltas = [
            DocumentDelta(doc_id="doc1", operation=OperationType.ADD, content="Doc 1", metadata={}, version=1),
            DocumentDelta(doc_id="doc2", operation=OperationType.ADD, content="Doc 2", metadata={}, version=1),
            DocumentDelta(doc_id="doc3", operation=OperationType.ADD, content="Doc 3", metadata={}, version=1),
        ]
        results = self.indexer.apply_deltas(deltas)

        assert len(results["success"]) == 3
        assert len(results["failed"]) == 0
        assert self.indexer.get_stats()["active_documents"] == 3

    def test_batch_deltas_with_conflicts(self):
        """Test batch with some conflicts."""
        add_delta = DocumentDelta(doc_id="doc1", operation=OperationType.ADD, content="Original", metadata={}, version=1)
        self.indexer.apply_delta(add_delta)

        deltas = [
            DocumentDelta(doc_id="doc2", operation=OperationType.ADD, content="New doc", metadata={}, version=1),
            DocumentDelta(doc_id="doc1", operation=OperationType.ADD, content="Duplicate", metadata={}, version=1),
        ]
        results = self.indexer.apply_deltas(deltas)

        assert len(results["success"]) == 1
        assert len(results["failed"]) == 1
        assert results["failed"][0] == "doc1"
        assert len(results["conflicts"]) == 1

    def test_audit_log(self):
        """Test audit event logging."""
        add_delta = DocumentDelta(doc_id="doc1", operation=OperationType.ADD, content="Audit test", metadata={}, version=1)
        self.indexer.apply_delta(add_delta)

        update_delta = DocumentDelta(doc_id="doc1", operation=OperationType.UPDATE, content="Updated", metadata={}, version=2)
        self.indexer.apply_delta(update_delta)

        delete_delta = DocumentDelta(doc_id="doc1", operation=OperationType.DELETE, version=3)
        self.indexer.apply_delta(delete_delta)

        audit_log = self.indexer.get_audit_log()
        assert len(audit_log) == 3

        add_event = next(e for e in audit_log if e["operation"] == "add")
        assert add_event["success"] is True
        assert add_event["doc_id"] == "doc1"
        assert add_event["version_after"] == 1

        update_event = next(e for e in audit_log if e["operation"] == "update")
        assert update_event["success"] is True
        assert update_event["version_before"] == 1
        assert update_event["version_after"] == 2

        delete_event = next(e for e in audit_log if e["operation"] == "delete")
        assert delete_event["success"] is True
        assert delete_event["version_before"] == 2
        assert delete_event["version_after"] == 3

    def test_audit_log_filter_by_doc_id(self):
        """Test filtering audit log by document ID."""
        self.indexer.apply_delta(DocumentDelta(doc_id="doc1", operation=OperationType.ADD, content="Doc 1", metadata={}, version=1))
        self.indexer.apply_delta(DocumentDelta(doc_id="doc2", operation=OperationType.ADD, content="Doc 2", metadata={}, version=1))
        self.indexer.apply_delta(DocumentDelta(doc_id="doc1", operation=OperationType.UPDATE, content="Doc 1 updated", metadata={}, version=2))

        doc1_log = self.indexer.get_audit_log(doc_id="doc1")
        assert len(doc1_log) == 2
        assert all(e["doc_id"] == "doc1" for e in doc1_log)

        doc2_log = self.indexer.get_audit_log(doc_id="doc2")
        assert len(doc2_log) == 1

    def test_checkpoint_creation(self):
        """Test checkpoint creation."""
        self.indexer.apply_delta(DocumentDelta(doc_id="doc1", operation=OperationType.ADD, content="Doc 1", metadata={}, version=1))
        self.indexer.apply_delta(DocumentDelta(doc_id="doc2", operation=OperationType.ADD, content="Doc 2", metadata={}, version=1))

        checkpoint = self.indexer.create_checkpoint(metadata={"reason": "test checkpoint"})
        assert checkpoint.document_count == 2
        assert checkpoint.state_hash is not None
        assert len(self.indexer._checkpoints) == 1

    def test_checkpoint_listing(self):
        """Test listing checkpoints."""
        self.indexer.create_checkpoint(metadata={"name": "cp1"})
        self.indexer.create_checkpoint(metadata={"name": "cp2"})

        checkpoints = self.indexer.get_checkpoints()
        assert len(checkpoints) == 2
        assert checkpoints[0]["metadata"]["name"] == "cp1"
        assert checkpoints[1]["metadata"]["name"] == "cp2"

    def test_invalidation_events(self):
        """Test invalidation event emission."""
        self.indexer.apply_delta(DocumentDelta(doc_id="doc1", operation=OperationType.ADD, content="Doc 1", metadata={}, version=1))
        self.indexer.apply_delta(DocumentDelta(doc_id="doc2", operation=OperationType.ADD, content="Doc 2", metadata={}, version=1))

        self.indexer.invalidate_documents({"doc1", "doc2"}, "expired")

        events = self.indexer.get_invalidation_events()
        assert len(events) == 1
        assert events[0]["reason"] == "expired"
        assert set(events[0]["doc_ids"]) == {"doc1", "doc2"}

        assert self.indexer.get_document("doc1") is None
        assert self.indexer.get_document("doc2") is None

    def test_search_functionality(self):
        """Test search returns results."""
        self.indexer.apply_delta(DocumentDelta(doc_id="doc1", operation=OperationType.ADD, content="Machine learning algorithms", metadata={"domain": "ai"}, version=1))
        self.indexer.apply_delta(DocumentDelta(doc_id="doc2", operation=OperationType.ADD, content="Deep learning neural networks", metadata={"domain": "ai"}, version=1))
        self.indexer.apply_delta(DocumentDelta(doc_id="doc3", operation=OperationType.ADD, content="Database optimization techniques", metadata={"domain": "db"}, version=1))

        results = self.indexer.search("machine learning", top_k=2)
        assert len(results) <= 2
        assert all("score" in r for r in results)

    def test_graph_adapter_integration(self):
        """Test graph adapter is updated on document operations."""
        add_delta = DocumentDelta(
            doc_id="doc1",
            operation=OperationType.ADD,
            content="Graph test content",
            metadata={"title": "Graph Doc", "domain": "test", "tags": ["tag1", "tag2"]},
            version=1,
        )
        self.indexer.apply_delta(add_delta)

        entities = self.indexer._graph_adapter.get_entities("doc1")
        assert "Graph Doc" in entities
        assert "tag1" in entities
        assert "tag2" in entities

    def test_vector_adapter_integration(self):
        """Test vector adapter is updated on document operations."""
        add_delta = DocumentDelta(doc_id="doc1", operation=OperationType.ADD, content="Vector test", metadata={}, version=1)
        self.indexer.apply_delta(add_delta)

        stored = self.indexer._vector_adapter.get("doc1")
        assert stored is not None
        assert stored["payload"]["content"] == "Vector test"

    def test_clear(self):
        """Test clearing the indexer."""
        self.indexer.apply_delta(DocumentDelta(doc_id="doc1", operation=OperationType.ADD, content="Test", metadata={}, version=1))
        self.indexer.apply_delta(DocumentDelta(doc_id="doc2", operation=OperationType.ADD, content="Test 2", metadata={}, version=1))

        self.indexer.clear()

        stats = self.indexer.get_stats()
        assert stats["total_documents"] == 0
        assert stats["active_documents"] == 0
        assert stats["audit_events"] == 0
        assert stats["checkpoints"] == 0
        assert stats["invalidation_events"] == 0


class TestConcurrency:
    """Test thread safety."""

    def test_concurrent_adds(self):
        """Test concurrent document additions."""
        indexer = IncrementalKnowledgeIndexer()
        indexer.initialize()
        errors = []

        def add_docs(thread_id: int):
            try:
                for i in range(10):
                    doc_id = f"doc_{thread_id}_{i}"
                    delta = DocumentDelta(doc_id=doc_id, operation=OperationType.ADD, content=f"Content {doc_id}", metadata={}, version=1)
                    indexer.apply_delta(delta)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_docs, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert indexer.get_stats()["active_documents"] == 50

    def test_concurrent_updates(self):
        """Test concurrent updates to same document."""
        indexer = IncrementalKnowledgeIndexer(conflict_strategy=ConflictStrategy.OVERWRITE)
        indexer.initialize()

        indexer.apply_delta(DocumentDelta(doc_id="shared", operation=OperationType.ADD, content="Initial", metadata={}, version=1))

        errors = []
        results = []

        def update_doc(thread_id: int):
            try:
                delta = DocumentDelta(
                    doc_id="shared",
                    operation=OperationType.UPDATE,
                    content=f"Update from thread {thread_id}",
                    metadata={},
                    version=thread_id + 2,
                )
                result = indexer.apply_delta(delta)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_doc, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        doc = indexer.get_document("shared")
        assert doc is not None
        assert doc.version >= 2


class TestDocumentDelta:
    """Test DocumentDelta dataclass."""

    def test_content_hash_auto_compute(self):
        """Test content hash is auto-computed."""
        delta = DocumentDelta(doc_id="doc1", operation=OperationType.ADD, content="Test content", metadata={}, version=1)
        assert delta.content_hash is not None
        assert len(delta.content_hash) == 16

    def test_explicit_content_hash(self):
        """Test explicit content hash is preserved."""
        delta = DocumentDelta(
            doc_id="doc1",
            operation=OperationType.ADD,
            content="Test content",
            metadata={},
            version=1,
            content_hash="explicit_hash_123",
        )
        assert delta.content_hash == "explicit_hash_123"

    def test_timestamp_auto_set(self):
        """Test timestamp is auto-set."""
        before = time.time()
        delta = DocumentDelta(doc_id="doc1", operation=OperationType.ADD, content="Test", metadata={}, version=1)
        after = time.time()
        assert before <= delta.timestamp <= after


class TestAdapters:
    """Test adapter implementations."""

    def test_in_memory_vector_adapter(self):
        """Test InMemoryVectorAdapter basic operations."""
        adapter = InMemoryVectorAdapter()

        vector = [0.1, 0.2, 0.3, 0.4]
        payload = {"id": "test", "content": "Test"}
        assert adapter.upsert("test", vector, payload) is True

        stored = adapter.get("test")
        assert stored is not None
        assert stored["vector"] == vector
        assert stored["payload"] == payload

        results = adapter.search([0.1, 0.2, 0.3, 0.4], top_k=5)
        assert len(results) == 1
        assert results[0]["id"] == "test"

        assert adapter.delete("test") is True
        assert adapter.get("test") is None
        assert adapter.size() == 0

    def test_in_memory_graph_adapter(self):
        """Test InMemoryGraphAdapter basic operations."""
        adapter = InMemoryGraphAdapter()

        adapter.add_document("doc1", "Title 1", "Content 1", "domain1", ["entity1", "entity2"])
        adapter.add_document("doc2", "Title 2", "Content 2", "domain1", ["entity2", "entity3"])

        entities = adapter.get_entities("doc1")
        assert "entity1" in entities
        assert "entity2" in entities

        adapter.remove_document("doc1")
        assert adapter.get_entities("doc1") == []

    def test_vector_adapter_search_scoring(self):
        """Test vector search returns cosine similarity scores."""
        adapter = InMemoryVectorAdapter()

        adapter.upsert("doc1", [1.0, 0.0, 0.0], {"id": "doc1"})
        adapter.upsert("doc2", [0.0, 1.0, 0.0], {"id": "doc2"})
        adapter.upsert("doc3", [0.707, 0.707, 0.0], {"id": "doc3"})

        results = adapter.search([1.0, 0.0, 0.0], top_k=3)
        assert results[0]["id"] == "doc1"
        assert results[0]["score"] > results[1]["score"]
        assert results[1]["id"] == "doc3"
        assert results[2]["id"] == "doc2"

    def test_custom_adapters(self):
        """Test using custom adapter implementations."""
        mock_vector = Mock(spec=VectorStoreAdapter)
        mock_graph = Mock(spec=GraphStoreAdapter)

        indexer = IncrementalKnowledgeIndexer(vector_adapter=mock_vector, graph_adapter=mock_graph)
        indexer.initialize()

        delta = DocumentDelta(doc_id="doc1", operation=OperationType.ADD, content="Test", metadata={}, version=1)
        indexer.apply_delta(delta)

        mock_vector.upsert.assert_called_once()
        mock_graph.add_document.assert_called_once()


class TestRollback:
    """Test rollback functionality."""

    def setup_method(self):
        self.indexer = IncrementalKnowledgeIndexer()
        self.indexer.initialize()

    def teardown_method(self):
        self.indexer.clear()

    def test_rollback_to_checkpoint(self):
        """Test rollback restores state."""
        self.indexer.apply_delta(DocumentDelta(doc_id="doc1", operation=OperationType.ADD, content="Doc 1", metadata={}, version=1))
        self.indexer.apply_delta(DocumentDelta(doc_id="doc2", operation=OperationType.ADD, content="Doc 2", metadata={}, version=1))

        checkpoint = self.indexer.create_checkpoint()

        self.indexer.apply_delta(DocumentDelta(doc_id="doc3", operation=OperationType.ADD, content="Doc 3", metadata={}, version=1))
        assert self.indexer.get_stats()["active_documents"] == 3

        self.indexer.rollback_to_checkpoint(checkpoint.checkpoint_id)

        assert self.indexer.get_stats()["active_documents"] == 2
        assert self.indexer.get_document("doc1") is not None
        assert self.indexer.get_document("doc2") is not None
        assert self.indexer.get_document("doc3") is None

    def test_rollback_nonexistent_checkpoint(self):
        """Test rollback to nonexistent checkpoint fails."""
        result = self.indexer.rollback_to_checkpoint("nonexistent")
        assert result is False


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        self.indexer = IncrementalKnowledgeIndexer()
        self.indexer.initialize()

    def teardown_method(self):
        self.indexer.clear()

    def test_add_without_content_fails(self):
        """Test ADD without content fails."""
        delta = DocumentDelta(doc_id="doc1", operation=OperationType.ADD, content=None, metadata={}, version=1)
        result = self.indexer.apply_delta(delta)
        assert result is False

    def test_update_nonexistent_fails(self):
        """Test UPDATE on nonexistent document fails."""
        delta = DocumentDelta(doc_id="nonexistent", operation=OperationType.UPDATE, content="New", metadata={}, version=1)
        result = self.indexer.apply_delta(delta)
        assert result is False

    def test_delete_nonexistent_fails(self):
        """Test DELETE on nonexistent document fails."""
        delta = DocumentDelta(doc_id="nonexistent", operation=OperationType.DELETE, version=1)
        result = self.indexer.apply_delta(delta)
        assert result is False

    def test_update_without_content_fails(self):
        """Test UPDATE without content fails."""
        self.indexer.apply_delta(DocumentDelta(doc_id="doc1", operation=OperationType.ADD, content="Original", metadata={}, version=1))

        delta = DocumentDelta(doc_id="doc1", operation=OperationType.UPDATE, content=None, metadata={}, version=2)
        result = self.indexer.apply_delta(delta)
        assert result is False

    def test_get_stats(self):
        """Test get_stats returns correct counts."""
        self.indexer.apply_delta(DocumentDelta(doc_id="doc1", operation=OperationType.ADD, content="Doc 1", metadata={}, version=1))
        self.indexer.apply_delta(DocumentDelta(doc_id="doc2", operation=OperationType.ADD, content="Doc 2", metadata={}, version=1))
        self.indexer.apply_delta(DocumentDelta(doc_id="doc1", operation=OperationType.DELETE, version=2))

        stats = self.indexer.get_stats()
        assert stats["total_documents"] == 2
        assert stats["active_documents"] == 1
        assert stats["tombstones"] == 1
        assert stats["vector_store_size"] == 1

    def test_audit_log_max_size(self):
        """Test audit log respects max size limit."""
        indexer = IncrementalKnowledgeIndexer(max_audit_events=5)
        indexer.initialize()

        for i in range(10):
            indexer.apply_delta(DocumentDelta(doc_id=f"doc{i}", operation=OperationType.ADD, content=f"Doc {i}", metadata={}, version=1))

        audit_log = indexer.get_audit_log(limit=20)
        assert len(audit_log) == 5

    def test_invalidation_with_metadata(self):
        """Test invalidation includes metadata."""
        self.indexer.apply_delta(DocumentDelta(doc_id="doc1", operation=OperationType.ADD, content="Doc 1", metadata={}, version=1))

        self.indexer.invalidate_documents({"doc1"}, "policy_violation", {"policy": "PII", "severity": "high"})

        events = self.indexer.get_invalidation_events()
        assert events[0]["metadata"]["policy"] == "PII"
        assert events[0]["metadata"]["severity"] == "high"


class TestAuditEventSerialization:
    """Test audit event serialization."""

    def test_audit_event_to_dict(self):
        """Test AuditEvent.to_dict() produces correct structure."""
        event = AuditEvent(
            event_id="test-id",
            timestamp=1234567890.0,
            operation=OperationType.ADD,
            doc_id="doc1",
            content_hash_before=None,
            content_hash_after="abc123",
            version_before=0,
            version_after=1,
            success=True,
            error_message=None,
            metadata={"source": "test"},
        )

        data = event.to_dict()
        assert data["event_id"] == "test-id"
        assert data["timestamp"] == 1234567890.0
        assert data["operation"] == "add"
        assert data["doc_id"] == "doc1"
        assert data["content_hash_before"] is None
        assert data["content_hash_after"] == "abc123"
        assert data["version_before"] == 0
        assert data["version_after"] == 1
        assert data["success"] is True
        assert data["error_message"] is None
        assert data["metadata"] == {"source": "test"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])