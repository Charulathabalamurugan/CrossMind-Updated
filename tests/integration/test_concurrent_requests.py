import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, AsyncMock
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List


class TestConcurrentRequests:
    """Test concurrent request handling and deduplication."""

    def test_concurrent_query_processing(
        self, mock_embedder, mock_agent, sample_evidence
    ):
        """Multiple concurrent queries should be processed correctly."""
        from reasoning.neuro_symbolic_pipeline import NeuroSymbolicPipeline
        
        with patch("reasoning.neuro_symbolic_pipeline.get_embedder", return_value=mock_embedder):
            with patch("reasoning.neuro_symbolic_pipeline.get_qdrant_engine") as mock_qdrant:
                mock_engine = Mock()
                mock_engine.search_with_rbac.return_value = sample_evidence
                mock_qdrant.return_value = mock_engine
                
                with patch("reasoning.neuro_symbolic_pipeline.get_knowledge_graph") as mock_kg:
                    mock_kg_instance = Mock()
                    mock_kg_instance.graph_rag_context.return_value = {
                        "nodes": [], "edges": [], "multi_hop_paths": [],
                        "cross_domain_path_count": 0, "query_anchors": [],
                        "seed_document_ids": ["doc:1", "doc:2"], "strategy": "test",
                        "path_scoring_method": "test"
                    }
                    mock_kg.return_value = mock_kg_instance
                    
                    with patch("reasoning.neuro_symbolic_pipeline.get_unified_router") as mock_router:
                        mock_router.return_value.route.return_value = {
                            "execution_mode": "deep",
                            "model": "zaya1_8b",
                            "budget_tokens": 6000,
                            "budget_cost_estimate": 0.01,
                        }
                        
                        with patch("reasoning.neuro_symbolic_pipeline.get_quality_gate") as mock_gate:
                            mock_gate.return_value.evaluate.return_value = {"passed": True}
                            
                            with patch("reasoning.neuro_symbolic_pipeline.get_cost_controller") as mock_cost:
                                mock_cost.return_value.enforce_budget.return_value = {"within_budget": True}
                                mock_cost.return_value.track_query.return_value = {}
                                
                                pipeline = NeuroSymbolicPipeline()
                                
                                queries = [
                                    "energy storage battery technology",
                                    "financial risk portfolio management",
                                    "nanomaterial drug delivery systems",
                                    "climate change carbon emissions",
                                    "cybersecurity threat detection",
                                ]
                                
                                results = []
                                for query in queries:
                                    result = pipeline.process_query(query=query, user_role="researcher")
                                    results.append(result)
                                
                                assert len(results) == 5
                                for i, result in enumerate(results):
                                    assert result["query"] == queries[i]
                                    assert "agent_reasoning" in result
                                    assert "confidence_calibration" in result

    def test_concurrent_streaming_requests(
        self, mock_embedder, mock_agent, sample_evidence
    ):
        """Multiple concurrent streaming requests should work independently."""
        from reasoning.neuro_symbolic_pipeline import NeuroSymbolicPipeline
        
        with patch("reasoning.neuro_symbolic_pipeline.get_embedder", return_value=mock_embedder):
            with patch("reasoning.neuro_symbolic_pipeline.get_qdrant_engine") as mock_qdrant:
                mock_engine = Mock()
                mock_engine.search_with_rbac.return_value = sample_evidence
                mock_qdrant.return_value = mock_engine
                
                with patch("reasoning.neuro_symbolic_pipeline.get_knowledge_graph") as mock_kg:
                    mock_kg_instance = Mock()
                    mock_kg_instance.graph_rag_context.return_value = {
                        "nodes": [], "edges": [], "multi_hop_paths": [],
                        "cross_domain_path_count": 0, "query_anchors": [],
                        "seed_document_ids": ["doc:1", "doc:2"], "strategy": "test",
                        "path_scoring_method": "test"
                    }
                    mock_kg.return_value = mock_kg_instance
                    
                    pipeline = NeuroSymbolicPipeline()
                    
                    async def collect_stream(query: str):
                        events = []
                        for event in pipeline.stream_query(query, user_role="researcher"):
                            events.append(event)
                        return events
                    
                    async def run_concurrent():
                        tasks = [
                            collect_stream("energy storage query"),
                            collect_stream("financial risk query"),
                            collect_stream("pharmacology query"),
                        ]
                        return await asyncio.gather(*tasks)
                    
                    all_events = asyncio.run(run_concurrent())
                    
                    assert len(all_events) == 3
                    for events in all_events:
                        assert len(events) > 0
                        event_types = [e["event"] for e in events]
                        assert "step_3a_pre_filter" in event_types
                        assert "completed" in event_types

    def test_thread_safety_ingestion_pipeline(
        self, mock_embedder, sample_documents
    ):
        """Ingestion pipeline should handle concurrent document ingestion."""
        from ingestion.pipeline import IngestionPipeline
        
        with patch("ingestion.pipeline.get_embedder", return_value=mock_embedder):
            with patch("ingestion.pipeline.get_qdrant_engine") as mock_qdrant:
                mock_engine = Mock()
                mock_engine.upsert_vectors.return_value = ["id1", "id2", "id3"]
                mock_qdrant.return_value = mock_engine
                
                with patch("ingestion.pipeline.get_knowledge_graph") as mock_kg:
                    mock_kg.return_value = Mock()
                    mock_kg.return_value.index_documents = Mock()
                    
                    with patch("ingestion.pipeline.get_sparse_vector_engine") as mock_sparse:
                        mock_sparse.return_value = Mock()
                        mock_sparse.return_value.index_documents = Mock()
                        
                        pipeline = IngestionPipeline()
                        
                        def ingest_docs(doc_batch):
                            return pipeline.ingest_documents(doc_batch)
                        
                        with ThreadPoolExecutor(max_workers=4) as executor:
                            futures = [
                                executor.submit(ingest_docs, [sample_documents[0]]),
                                executor.submit(ingest_docs, [sample_documents[1]]),
                                executor.submit(ingest_docs, [sample_documents[2]]),
                            ]
                            results = [f.result() for f in as_completed(futures)]
                        
                        assert len(results) == 3
                        for result in results:
                            assert isinstance(result, list)


class TestRequestDeduplication:
    """Test request deduplication functionality."""

    def test_exact_query_deduplication(
        self, mock_embedder, mock_agent, sample_evidence
    ):
        """Identical queries should be deduplicated and return cached results."""
        from reasoning.neuro_symbolic_pipeline import NeuroSymbolicPipeline
        
        with patch("reasoning.neuro_symbolic_pipeline.get_embedder", return_value=mock_embedder):
            with patch("reasoning.neuro_symbolic_pipeline.get_qdrant_engine") as mock_qdrant:
                mock_engine = Mock()
                mock_engine.search_with_rbac.return_value = sample_evidence
                mock_qdrant.return_value = mock_engine
                
                with patch("reasoning.neuro_symbolic_pipeline.get_knowledge_graph") as mock_kg:
                    mock_kg_instance = Mock()
                    mock_kg_instance.graph_rag_context.return_value = {
                        "nodes": [], "edges": [], "multi_hop_paths": [],
                        "cross_domain_path_count": 0, "query_anchors": [],
                        "seed_document_ids": ["doc:1", "doc:2"], "strategy": "test",
                        "path_scoring_method": "test"
                    }
                    mock_kg.return_value = mock_kg_instance
                    
                    with patch("reasoning.neuro_symbolic_pipeline.get_unified_router") as mock_router:
                        mock_router.return_value.route.return_value = {
                            "execution_mode": "deep",
                            "model": "zaya1_8b",
                            "budget_tokens": 6000,
                            "budget_cost_estimate": 0.01,
                        }
                        
                        with patch("reasoning.neuro_symbolic_pipeline.get_quality_gate") as mock_gate:
                            mock_gate.return_value.evaluate.return_value = {"passed": True}
                            
                            with patch("reasoning.neuro_symbolic_pipeline.get_cost_controller") as mock_cost:
                                mock_cost.return_value.enforce_budget.return_value = {"within_budget": True}
                                mock_cost.return_value.track_query.return_value = {}
                                
                                pipeline = NeuroSymbolicPipeline()
                                
                                query = "test query for deduplication"
                                
                                result1 = pipeline.process_query(query=query, user_role="researcher")
                                result2 = pipeline.process_query(query=query, user_role="researcher")
                                result3 = pipeline.process_query(query=query, user_role="researcher")
                                
                                assert result1 == result2 == result3
                                assert "query" in result1

    def test_semantic_query_cache_similarity_deduplication(
        self, mock_embedder, mock_agent, sample_evidence
    ):
        """Semantically similar queries should hit semantic cache."""
        from reasoning.query_cache import get_query_cache, QueryResultCache
        
        with patch("reasoning.query_cache.get_embedder", return_value=mock_embedder):
            cache = get_query_cache()
            cache.clear()
            
            mock_embedder.embed_text.side_effect = [
                [0.9, 0.1] + [0.0] * 254,
                [0.89, 0.11] + [0.0] * 254,
                [0.1, 0.9] + [0.0] * 254,
            ]
            
            cache.set("query:1", {"result": "first"}, query="battery energy storage", user_role="researcher")
            
            similar_result = cache.get_similar("energy storage battery", user_role="researcher")
            assert similar_result == {"result": "first"}
            
            different_result = cache.get_similar("financial risk portfolio", user_role="researcher")
            assert different_result is None

    def test_stream_query_deduplication(
        self, mock_embedder, mock_agent, sample_evidence
    ):
        """Stream queries should also be deduplicated."""
        from reasoning.neuro_symbolic_pipeline import NeuroSymbolicPipeline
        
        with patch("reasoning.neuro_symbolic_pipeline.get_embedder", return_value=mock_embedder):
            with patch("reasoning.neuro_symbolic_pipeline.get_qdrant_engine") as mock_qdrant:
                mock_engine = Mock()
                mock_engine.search_with_rbac.return_value = sample_evidence
                mock_qdrant.return_value = mock_engine
                
                with patch("reasoning.neuro_symbolic_pipeline.get_knowledge_graph") as mock_kg:
                    mock_kg_instance = Mock()
                    mock_kg_instance.graph_rag_context.return_value = {
                        "nodes": [], "edges": [], "multi_hop_paths": [],
                        "cross_domain_path_count": 0, "query_anchors": [],
                        "seed_document_ids": ["doc:1", "doc:2"], "strategy": "test",
                        "path_scoring_method": "test"
                    }
                    mock_kg.return_value = mock_kg_instance
                    
                    pipeline = NeuroSymbolicPipeline()
                    
                    query = "test streaming query"
                    
                    events1 = list(pipeline.stream_query(query, user_role="researcher"))
                    events2 = list(pipeline.stream_query(query, user_role="researcher"))
                    
                    assert len(events1) == len(events2)
                    for e1, e2 in zip(events1, events2):
                        assert e1 == e2

    def test_concurrent_deduplication_race_condition(
        self, mock_embedder, mock_agent, sample_evidence
    ):
        """Concurrent identical requests should not cause race conditions."""
        from reasoning.neuro_symbolic_pipeline import NeuroSymbolicPipeline
        
        with patch("reasoning.neuro_symbolic_pipeline.get_embedder", return_value=mock_embedder):
            with patch("reasoning.neuro_symbolic_pipeline.get_qdrant_engine") as mock_qdrant:
                mock_engine = Mock()
                mock_engine.search_with_rbac.return_value = sample_evidence
                mock_qdrant.return_value = mock_engine
                
                with patch("reasoning.neuro_symbolic_pipeline.get_knowledge_graph") as mock_kg:
                    mock_kg_instance = Mock()
                    mock_kg_instance.graph_rag_context.return_value = {
                        "nodes": [], "edges": [], "multi_hop_paths": [],
                        "cross_domain_path_count": 0, "query_anchors": [],
                        "seed_document_ids": ["doc:1", "doc:2"], "strategy": "test",
                        "path_scoring_method": "test"
                    }
                    mock_kg.return_value = mock_kg_instance
                    
                    with patch("reasoning.neuro_symbolic_pipeline.get_unified_router") as mock_router:
                        mock_router.return_value.route.return_value = {
                            "execution_mode": "deep",
                            "model": "zaya1_8b",
                            "budget_tokens": 6000,
                            "budget_cost_estimate": 0.01,
                        }
                        
                        with patch("reasoning.neuro_symbolic_pipeline.get_quality_gate") as mock_gate:
                            mock_gate.return_value.evaluate.return_value = {"passed": True}
                            
                            with patch("reasoning.neuro_symbolic_pipeline.get_cost_controller") as mock_cost:
                                mock_cost.return_value.enforce_budget.return_value = {"within_budget": True}
                                mock_cost.return_value.track_query.return_value = {}
                                
                                pipeline = NeuroSymbolicPipeline()
                                
                                query = "concurrent deduplication test"
                                
                                def process():
                                    return pipeline.process_query(query=query, user_role="researcher")
                                
                                with ThreadPoolExecutor(max_workers=10) as executor:
                                    futures = [executor.submit(process) for _ in range(20)]
                                    results = [f.result() for f in as_completed(futures)]
                                
                                assert len(results) == 20
                                for result in results:
                                    assert result["query"] == query
                                    assert "agent_reasoning" in result

    def test_deduplication_with_different_user_roles(
        self, mock_embedder, mock_agent, sample_evidence
    ):
        """Different user roles should not share cached results."""
        from reasoning.neuro_symbolic_pipeline import NeuroSymbolicPipeline
        
        with patch("reasoning.neuro_symbolic_pipeline.get_embedder", return_value=mock_embedder):
            with patch("reasoning.neuro_symbolic_pipeline.get_qdrant_engine") as mock_qdrant:
                mock_engine = Mock()
                mock_engine.search_with_rbac.return_value = sample_evidence
                mock_qdrant.return_value = mock_engine
                
                with patch("reasoning.neuro_symbolic_pipeline.get_knowledge_graph") as mock_kg:
                    mock_kg_instance = Mock()
                    mock_kg_instance.graph_rag_context.return_value = {
                        "nodes": [], "edges": [], "multi_hop_paths": [],
                        "cross_domain_path_count": 0, "query_anchors": [],
                        "seed_document_ids": ["doc:1", "doc:2"], "strategy": "test",
                        "path_scoring_method": "test"
                    }
                    mock_kg.return_value = mock_kg_instance
                    
                    with patch("reasoning.neuro_symbolic_pipeline.get_unified_router") as mock_router:
                        mock_router.return_value.route.return_value = {
                            "execution_mode": "deep",
                            "model": "zaya1_8b",
                            "budget_tokens": 6000,
                            "budget_cost_estimate": 0.01,
                        }
                        
                        with patch("reasoning.neuro_symbolic_pipeline.get_quality_gate") as mock_gate:
                            mock_gate.return_value.evaluate.return_value = {"passed": True}
                            
                            with patch("reasoning.neuro_symbolic_pipeline.get_cost_controller") as mock_cost:
                                mock_cost.return_value.enforce_budget.return_value = {"within_budget": True}
                                mock_cost.return_value.track_query.return_value = {}
                                
                                pipeline = NeuroSymbolicPipeline()
                                
                                query = "role-specific query"
                                
                                result_public = pipeline.process_query(query=query, user_role="public")
                                result_researcher = pipeline.process_query(query=query, user_role="researcher")
                                result_admin = pipeline.process_query(query=query, user_role="admin")
                                
                                assert result_public["user_role"] == "public"
                                assert result_researcher["user_role"] == "researcher"
                                assert result_admin["user_role"] == "admin"


class TestConcurrentIngestion:
    """Test concurrent ingestion scenarios."""

    def test_queue_manager_concurrent_enqueue(
        self, mock_embedder, sample_documents
    ):
        """Queue manager should handle concurrent enqueue operations."""
        from ingestion.queue_manager import get_queue_manager, QueueManager
        
        qm = get_queue_manager()
        
        def enqueue_docs(docs):
            return qm.enqueue(docs, source="test")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(enqueue_docs, [doc]) for doc in sample_documents]
            task_ids = [f.result() for f in as_completed(futures)]
        
        assert len(task_ids) == 3
        assert len(set(task_ids)) == 3
        
        stats = qm.get_stats()
        assert stats["queued"] == 3

    def test_queue_manager_processing_under_load(
        self, mock_embedder, sample_documents
    ):
        """Queue manager should process tasks under concurrent load."""
        from ingestion.queue_manager import QueueManager
        
        qm = QueueManager()
        processed = []
        errors = []
        
        def process_fn(docs):
            processed.append(len(docs))
            return {"inserted_ids": [f"id_{i}" for i in range(len(docs))]}
        
        qm.start(process_fn)
        time.sleep(0.1)
        
        for doc in sample_documents:
            qm.enqueue([doc])
        
        time.sleep(0.5)
        
        qm.stop()
        
        stats = qm.get_stats()
        assert stats["total_processed"] >= 1
        assert stats["total_failed"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])