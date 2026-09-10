import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List


class TestServiceFailureGracefulDegradation:
    """Test graceful degradation when external services fail."""

    def test_qdrant_failure_falls_back_to_memory_store(
        self, mock_embedder, mock_agent, sample_documents
    ):
        """When Qdrant is unavailable, system should use in-memory fallback."""
        from vector_store.qdrant_engine import get_qdrant_engine, QdrantVectorEngine
        
        with patch("vector_store.qdrant_engine.QdrantClient", side_effect=Exception("Connection refused")):
            engine = get_qdrant_engine()
            
            assert engine.client is None
            
            ids = engine.upsert_vectors(sample_documents)
            assert len(ids) == 3
            
            results = engine.search_with_rbac(
                query_vector=[0.1] * 256,
                user_role="researcher",
                top_k=5,
                query_text="battery"
            )
            assert isinstance(results, list)
            assert len(results) > 0
            for r in results:
                assert "id" in r
                assert "score" in r
                assert "payload" in r

    def test_qdrant_search_failure_falls_back_to_memory(
        self, mock_embedder, mock_agent, sample_documents
    ):
        """When Qdrant search fails, should fall back to memory search."""
        from vector_store.qdrant_engine import get_qdrant_engine
        
        with patch("vector_store.qdrant_engine.QdrantClient") as mock_client_class:
            mock_client = Mock()
            mock_client.get_collections.return_value = Mock(collections=[])
            mock_client.upsert.return_value = None
            mock_client.search.side_effect = Exception("Search failed")
            mock_client_class.return_value = mock_client
            
            engine = get_qdrant_engine()
            engine.upsert_vectors(sample_documents)
            
            results = engine.search_with_rbac(
                query_vector=[0.1] * 256,
                user_role="researcher",
                top_k=5,
                query_text="battery"
            )
            
            assert isinstance(results, list)
            assert len(results) > 0

    def test_redis_failure_uses_memory_cache(self, mock_embedder):
        """When Redis is unavailable, should use in-memory cache fallback."""
        from ingestion.redis_cache import get_redis_cache, RedisCache
        
        with patch("ingestion.redis_cache.redis", side_effect=ImportError("No redis module")):
            cache = get_redis_cache()
            
            assert cache._client is None
            
            cache.set("test_key", {"data": "value"})
            result = cache.get("test_key")
            assert result == {"data": "value"}
            
            cache.delete("test_key")
            assert cache.get("test_key") is None

    def test_neo4j_failure_uses_in_memory_fallback(self, mock_embedder):
        """When Neo4j is unavailable, should use in-memory graph fallback."""
        from reasoning.neo4j_graph import get_neo4j_store, Neo4jGraph
        
        with patch("reasoning.neo4j_graph.GraphDatabase", side_effect=ImportError("No neo4j module")):
            graph = get_neo4j_store()
            
            assert graph.driver is None
            
            docs = [
                {"id": "doc:1", "title": "Test Doc", "domain": "energy"}
            ]
            graph.index_documents(docs)
            
            assert "doc:1" in graph.nodes
            
            evidence = [{"id": "doc:1", "payload": {"title": "Test Doc", "domain": "energy"}}]
            result = graph.graph_rag_context(evidence, ["energy"])
            
            assert result["strategy"] == "in_memory_fallback"
            assert len(result["multi_hop_paths"]) > 0

    def test_embedding_service_failure_uses_deterministic_fallback(self):
        """When embedding service fails, should use deterministic vector generation."""
        from ingestion.embedding import get_embedder, Embedder
        
        embedder = Embedder()
        
        vec = embedder.embed_text("test query")
        assert isinstance(vec, list)
        assert len(vec) == settings.EMBEDDING_DIM
        
        import numpy as np
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.001

    def test_llm_failure_uses_lite_llm_fallback(
        self, mock_embedder, sample_evidence
    ):
        """When ZAYA1-8B fails, should fall back to LiteLLM."""
        from reasoning.neuro_symbolic_pipeline import NeuroSymbolicPipeline
        from config import settings
        
        settings.ZAYA1_8B_REASONING_ENABLED = False
        settings.LITELLM_ENABLED = True
        
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
                        "seed_document_ids": [], "strategy": "test",
                        "path_scoring_method": "test"
                    }
                    mock_kg.return_value = mock_kg_instance
                    
                    with patch("reasoning.neuro_symbolic_pipeline.ZAYA1_8BAgent") as mock_agent_class:
                        mock_agent = Mock()
                        mock_agent.reason_and_synthesize.side_effect = Exception("LLM unavailable")
                        mock_agent_class.return_value = mock_agent
                        
                        pipeline = NeuroSymbolicPipeline()
                        
                        with patch.object(pipeline, '_lite_llm_reasoning') as mock_lite:
                            mock_lite.return_value = {
                                "model": "lite-llm",
                                "think_block": "Lite reasoning",
                                "tool_calls": [],
                                "output_text": "Lite hypothesis",
                                "hypothesis": "Lite hypothesis",
                                "cited_evidence_ids": ["doc:1"],
                                "confidence_score": 0.75,
                            }
                            
                            result = pipeline.process_query(
                                query="test query",
                                user_role="researcher"
                            )
                            
                            assert result["agent_reasoning"]["model"] == "lite-llm"
                            mock_lite.assert_called_once()

    def test_all_services_fail_returns_degraded_response(
        self, mock_embedder, sample_documents
    ):
        """When all external services fail, should return a degraded but valid response."""
        from reasoning.neuro_symbolic_pipeline import NeuroSymbolicPipeline
        
        with patch("reasoning.neuro_symbolic_pipeline.get_embedder", return_value=mock_embedder):
            with patch("reasoning.neuro_symbolic_pipeline.get_qdrant_engine") as mock_qdrant:
                mock_engine = Mock()
                mock_engine.search_with_rbac.return_value = []
                mock_qdrant.return_value = mock_engine
                
                with patch("reasoning.neuro_symbolic_pipeline.get_knowledge_graph") as mock_kg:
                    mock_kg_instance = Mock()
                    mock_kg_instance.graph_rag_context.return_value = {
                        "nodes": [], "edges": [], "multi_hop_paths": [],
                        "cross_domain_path_count": 0, "query_anchors": [],
                        "seed_document_ids": [], "strategy": "empty",
                        "path_scoring_method": "none"
                    }
                    mock_kg.return_value = mock_kg_instance
                    
                    with patch("reasoning.neuro_symbolic_pipeline.ZAYA1_8BAgent") as mock_agent_class:
                        mock_agent = Mock()
                        mock_agent.reason_and_synthesize.side_effect = Exception("All services down")
                        mock_agent_class.return_value = mock_agent
                        
                        with patch("reasoning.neuro_symbolic_pipeline.get_unified_router") as mock_router:
                            mock_router.return_value.route.return_value = {
                                "execution_mode": "fast",
                                "model": "lite-llm",
                                "budget_tokens": 1500,
                                "budget_cost_estimate": 0.001,
                            }
                            
                            with patch("reasoning.neuro_symbolic_pipeline.get_quality_gate") as mock_gate:
                                mock_gate.return_value.evaluate.return_value = {"passed": True}
                                
                                with patch("reasoning.neuro_symbolic_pipeline.get_cost_controller") as mock_cost:
                                    mock_cost.return_value.enforce_budget.return_value = {"within_budget": True}
                                    mock_cost.return_value.track_query.return_value = {}
                                    
                                    pipeline = NeuroSymbolicPipeline()
                                    
                                    with patch.object(pipeline, '_lite_llm_reasoning') as mock_lite:
                                        mock_lite.return_value = {
                                            "model": "lite-llm",
                                            "think_block": "Degraded reasoning",
                                            "tool_calls": [],
                                            "output_text": "Degraded response",
                                            "hypothesis": "Degraded",
                                            "cited_evidence_ids": [],
                                            "confidence_score": 0.5,
                                        }
                                        
                                        result = pipeline.process_query(
                                            query="test query",
                                            user_role="researcher"
                                        )
                                        
                                        assert "query" in result
                                        assert "agent_reasoning" in result
                                        assert result["agent_reasoning"]["model"] == "lite-llm"
                                        assert result["confidence_calibration"]["calibrated_confidence"] >= 0.0


class TestServiceFailureScenarios:
    """Additional service failure scenarios."""

    def test_qdrant_upsert_failure_continues_with_memory(self, mock_embedder, mock_agent):
        """Upsert failure should not block memory store."""
        from vector_store.qdrant_engine import get_qdrant_engine
        
        with patch("vector_store.qdrant_engine.QdrantClient") as mock_client_class:
            mock_client = Mock()
            mock_client.get_collections.return_value = Mock(collections=[])
            mock_client.upsert.side_effect = Exception("Upsert failed")
            mock_client_class.return_value = mock_client
            
            engine = get_qdrant_engine()
            docs = [{"id": "doc:1", "vector": [0.1]*256, "payload": {"title": "Test", "content": "Content", "domain": "test"}}]
            
            ids = engine.upsert_vectors(docs)
            
            assert len(ids) == 1
            assert len(engine._memory_store) == 1

    def test_redis_ping_failure_returns_false(self):
        """Redis ping should return False when connection fails."""
        from ingestion.redis_cache import RedisCache
        
        with patch("ingestion.redis_cache.redis") as mock_redis:
            mock_client = Mock()
            mock_client.ping.side_effect = Exception("Connection failed")
            mock_redis.Redis.return_value = mock_client
            
            cache = RedisCache()
            assert cache.ping() is False

    def test_neo4j_session_failure_falls_back(self, mock_embedder):
        """Neo4j session failure should trigger in-memory fallback."""
        from reasoning.neo4j_graph import Neo4jGraph
        
        with patch("reasoning.neo4j_graph.GraphDatabase") as mock_neo4j:
            mock_driver = Mock()
            mock_session = Mock()
            mock_session.run.side_effect = Exception("Session failed")
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_neo4j.driver.return_value = mock_driver
            
            graph = Neo4jGraph()
            
            evidence = [{"id": "doc:1", "payload": {"title": "Test", "domain": "energy"}}]
            result = graph.graph_rag_context(evidence, ["energy"])
            
            assert result["strategy"] == "in_memory_fallback"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])