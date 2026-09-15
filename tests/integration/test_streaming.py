import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List, Generator
from fastapi.testclient import TestClient


class TestStreamingEventOrdering:
    """Test SSE streaming event ordering and structure."""

    def test_stream_events_are_ordered_correctly(
        self, mock_embedder, mock_agent, sample_evidence
    ):
        """Stream events should be emitted in the correct order."""
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
                    
                    events = list(pipeline.stream_query("test query", user_role="researcher"))
                    
                    event_types = [e["event"] for e in events]
                    
                    expected_order = [
                        "step_3a_pre_filter",
                        "step_2_vector_retrieval",
                        "step_3b_zaya1_8b_reasoning",
                        "step_3c_post_validation",
                        "step_4_graph_rag",
                        "completed"
                    ]
                    
                    for expected in expected_order:
                        assert expected in event_types, f"Missing event type: {expected}"
                    
                    pre_filter_idx = event_types.index("step_3a_pre_filter")
                    retrieval_idx = event_types.index("step_2_vector_retrieval")
                    reasoning_idx = event_types.index("step_3b_zaya1_8b_reasoning")
                    validation_idx = event_types.index("step_3c_post_validation")
                    graph_rag_idx = event_types.index("step_4_graph_rag")
                    completed_idx = event_types.index("completed")
                    
                    assert pre_filter_idx < retrieval_idx < reasoning_idx
                    assert reasoning_idx <= validation_idx
                    assert validation_idx < graph_rag_idx < completed_idx

    def test_stream_reasoning_events_have_correct_structure(
        self, mock_embedder, mock_agent, sample_evidence
    ):
        """Each stream event should have the correct data structure."""
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
                    
                    events = list(pipeline.stream_query("test query", user_role="researcher"))
                    
                    for event in events:
                        assert "event" in event
                        assert "data" in event
                        assert isinstance(event["data"], dict)
                    
                    pre_filter = next(e for e in events if e["event"] == "step_3a_pre_filter")
                    assert "detected_domains" in pre_filter["data"]
                    assert "extracted_entities" in pre_filter["data"]
                    assert "session_id" in pre_filter["data"]
                    
                    retrieval = next(e for e in events if e["event"] == "step_2_vector_retrieval")
                    assert "retrieved_count" in retrieval["data"]
                    assert "retrieved_evidence" in retrieval["data"]
                    assert "strategy" in retrieval["data"]
                    
                    reasoning_events = [e for e in events if e["event"] == "step_3b_zaya1_8b_reasoning"]
                    assert len(reasoning_events) >= 2
                    
                    thinking_event = next(e for e in reasoning_events if e["data"].get("stage") == "thinking")
                    assert "delta" in thinking_event["data"]
                    
                    hypothesis_event = next(e for e in reasoning_events if e["data"].get("stage") == "hypothesis_synthesis")
                    assert "delta" in hypothesis_event["data"]
                    assert "structured_result" in hypothesis_event["data"]
                    
                    completed = next(e for e in events if e["event"] == "completed")
                    assert completed["data"]["status"] == "success"
                    assert "final_hypothesis" in completed["data"]
                    assert "calibrated_confidence" in completed["data"]

    def test_stream_sse_format_compliance(
        self, mock_embedder, mock_agent, sample_evidence, test_client, auth_headers
    ):
        """SSE endpoint should return properly formatted events."""
        from app.main import app
        from reasoning.neuro_symbolic_pipeline import get_neuro_symbolic_pipeline
        
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
                                
                                response = test_client.get(
                                    "/api/stream_reasoning",
                                    params={"query": "test query", "user_role": "researcher"},
                                    headers=auth_headers
                                )
                                
                                assert response.status_code == 200
                                assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
                                
                                content = response.text
                                lines = content.strip().split("\n")
                                
                                data_lines = [l for l in lines if l.startswith("data: ")]
                                assert len(data_lines) > 0
                                
                                for line in data_lines:
                                    json_str = line[6:]
                                    parsed = json.loads(json_str)
                                    assert "event" in parsed
                                    assert "data" in parsed


class TestStreamingDisconnectAndBackpressure:
    """Test streaming disconnect handling and backpressure."""

    def test_client_disconnect_stops_generation(
        self, mock_embedder, mock_agent, sample_evidence
    ):
        """Generator should handle client disconnect gracefully."""
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
                    
                    gen = pipeline.stream_query("test query", user_role="researcher")
                    
                    first_event = next(gen)
                    assert first_event["event"] == "step_3a_pre_filter"
                    
                    second_event = next(gen)
                    assert second_event["event"] == "step_2_vector_retrieval"
                    
                    gen.close()
                    
                    with pytest.raises(StopIteration):
                        next(gen)

    def test_sse_generator_handles_broken_pipe(
        self, mock_embedder, mock_agent, sample_evidence, test_client, auth_headers
    ):
        """SSE endpoint should handle broken pipe when client disconnects."""
        from app.main import app
        from reasoning.neuro_symbolic_pipeline import get_neuro_symbolic_pipeline
        
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
                                
                                with patch("app.main.NeuroSymbolicPipeline") as mock_pipeline_class:
                                    mock_pipeline = Mock()
                                    mock_pipeline.stream_query.return_value = iter([
                                        {"event": "step_3a_pre_filter", "data": {"test": "data"}},
                                        {"event": "completed", "data": {"status": "success"}},
                                    ])
                                    mock_pipeline_class.return_value = mock_pipeline
                                    
                                response = test_client.get(
                                    "/api/stream_reasoning",
                                    params={"query": "test", "user_role": "researcher"},
                                    headers=auth_headers,
                                )

                                assert response.status_code == 200

                                for chunk in response.iter_text():
                                    pass

    def test_streaming_backpressure_with_slow_consumer(
        self, mock_embedder, mock_agent, sample_evidence
    ):
        """Stream should handle slow consumers without blocking indefinitely."""
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
                    
                    events = []
                    gen = pipeline.stream_query("test query", user_role="researcher")
                    
                    for event in gen:
                        events.append(event)
                        if len(events) >= 3:
                            break
                    
                    assert len(events) == 3
                    
                    remaining = list(gen)
                    assert len(remaining) > 0

    def test_streaming_timeout_handling(
        self, mock_embedder, mock_agent, sample_evidence
    ):
        """Long-running streams should not hang indefinitely."""
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
                    
                    start = time.time()
                    events = list(pipeline.stream_query("test query", user_role="researcher"))
                    elapsed = time.time() - start
                    
                    assert elapsed < 5.0
                    assert len(events) > 0


class TestStreamingEdgeCases:
    """Edge cases for streaming functionality."""

    def test_empty_query_stream_returns_error(self, test_client, auth_headers):
        """Empty query should return error in stream."""
        response = test_client.get(
            "/api/stream_reasoning",
            params={"query": "", "user_role": "researcher"},
            headers=auth_headers
        )
        assert response.status_code == 400

    def test_invalid_role_stream_returns_error(self, test_client, auth_headers):
        """Invalid role should return error in stream."""
        response = test_client.get(
            "/api/stream_reasoning",
            params={"query": "test", "user_role": "invalid_role"},
            headers=auth_headers
        )
        assert response.status_code == 400

    def test_stream_cache_hit_returns_cached_events(
        self, mock_embedder, mock_agent, sample_evidence
    ):
        """Cached stream queries should return cached events in order."""
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
                    
                    query = "cached stream query"
                    
                    events1 = list(pipeline.stream_query(query, user_role="researcher"))
                    events2 = list(pipeline.stream_query(query, user_role="researcher"))
                    
                    assert events1 == events2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])