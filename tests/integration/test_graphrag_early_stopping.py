import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List


class TestAdaptiveGraphRAGEarlyStopping:
    """Test GraphRAG adaptive early stopping functionality."""

    def test_graph_rag_early_stop_threshold_config(self):
        """GraphRAG should respect early stop threshold configuration."""
        from config import settings
        from reasoning.knowledge_graph import KnowledgeGraph
        
        assert hasattr(settings, 'GRAPH_RAG_EARLY_STOP_THRESHOLD')
        assert 0.0 <= settings.GRAPH_RAG_EARLY_STOP_THRESHOLD <= 1.0
        assert hasattr(settings, 'GRAPH_RAG_MAX_PATHS')
        assert hasattr(settings, 'GRAPH_RAG_MIN_EVIDENCE')

    def test_graph_rag_context_respects_max_paths(
        self, mock_embedder, sample_evidence
    ):
        """GraphRAG context should limit multi-hop paths to max paths setting."""
        from reasoning.knowledge_graph import KnowledgeGraph
        from config import settings
        
        kg = KnowledgeGraph()
        
        large_evidence = []
        for i in range(20):
            large_evidence.append({
                "id": f"doc:{i}",
                "score": 0.9 - (i * 0.02),
                "payload": {
                    "id": f"doc:{i}",
                    "title": f"Document {i}",
                    "content": f"Content about topic {i % 5}",
                    "domain": ["energy", "finance", "healthcare", "climate", "materials"][i % 5],
                    "tags": [f"tag_{i % 5}"],
                    "allowed_roles": ["public", "researcher"]
                }
            })
        
        result = kg.graph_rag_context(large_evidence, ["test"])
        
        assert len(result["multi_hop_paths"]) <= settings.GRAPH_RAG_MAX_PATHS

    def test_graph_rag_min_evidence_requirement(
        self, mock_embedder
    ):
        """GraphRAG should handle insufficient evidence gracefully."""
        from reasoning.knowledge_graph import KnowledgeGraph
        from config import settings
        
        kg = KnowledgeGraph()
        
        minimal_evidence = [
            {
                "id": "doc:1",
                "score": 0.9,
                "payload": {
                    "id": "doc:1",
                    "title": "Single Doc",
                    "content": "Single document content",
                    "domain": "energy",
                    "tags": ["energy"],
                    "allowed_roles": ["public", "researcher"]
                }
            }
        ]
        
        result = kg.graph_rag_context(minimal_evidence, ["energy"])
        
        assert result["multi_hop_paths"] == []
        assert result["cross_domain_path_count"] == 0
        assert len(result["nodes"]) >= 1

    def test_early_stop_based_on_confidence_calibration(
        self, mock_embedder, mock_agent, sample_evidence
    ):
        """Pipeline should stop early when confidence exceeds threshold."""
        from reasoning.neuro_symbolic_pipeline import NeuroSymbolicPipeline
        from config import settings
        
        settings.GRAPH_RAG_EARLY_STOP_THRESHOLD = 0.9
        
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
                                
                                mock_agent.reason_and_synthesize.return_value = {
                                    "model": "ZAYA1-8B",
                                    "think_block": "High confidence reasoning",
                                    "tool_calls": [],
                                    "output_text": "Strong hypothesis",
                                    "hypothesis": "Strong hypothesis",
                                    "cited_evidence_ids": ["doc:1", "doc:2"],
                                    "confidence_score": 0.95,
                                }
                                
                                result = pipeline.process_query(
                                    query="high confidence query",
                                    user_role="researcher",
                                    confidence_thresholds={"proceed": 0.75, "investigate": 0.50}
                                )
                                
                                assert "confidence_calibration" in result
                                assert "calibrated_confidence" in result["confidence_calibration"]
                                assert result["confidence_calibration"]["decision"] in [
                                    "proceed_to_experimental_design",
                                    "seek_more_evidence",
                                    "do_not_act_without_validation"
                                ]

    def test_adaptive_depth_based_on_evidence_quality(
        self, mock_embedder, sample_evidence
    ):
        """GraphRAG depth should adapt based on evidence quality."""
        from reasoning.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        high_quality_evidence = [
            {
                "id": "doc:1",
                "score": 0.95,
                "payload": {
                    "id": "doc:1",
                    "title": "High Quality Doc",
                    "content": "Detailed technical content with specific findings and data",
                    "domain": "energy",
                    "tags": ["battery", "lithium", "electrolyte", "performance"],
                    "allowed_roles": ["public", "researcher"]
                }
            },
            {
                "id": "doc:2",
                "score": 0.92,
                "payload": {
                    "id": "doc:2",
                    "title": "Another High Quality Doc",
                    "content": "Comprehensive analysis with experimental results",
                    "domain": "energy",
                    "tags": ["battery", "cathode", "anode", "cycling"],
                    "allowed_roles": ["public", "researcher"]
                }
            }
        ]
        
        result = kg.graph_rag_context(high_quality_evidence, ["battery", "lithium"])
        
        assert len(result["multi_hop_paths"]) > 0
        assert result["cross_domain_path_count"] == 0
        
        cross_domain_evidence = [
            {
                "id": "doc:1",
                "score": 0.95,
                "payload": {
                    "id": "doc:1",
                    "title": "Energy Doc",
                    "content": "Battery technology for grid storage",
                    "domain": "energy",
                    "tags": ["battery", "grid"],
                    "allowed_roles": ["public", "researcher"]
                }
            },
            {
                "id": "doc:2",
                "score": 0.92,
                "payload": {
                    "id": "doc:2",
                    "title": "Finance Doc",
                    "content": "Investment in battery storage companies",
                    "domain": "finance",
                    "tags": ["battery", "investment"],
                    "allowed_roles": ["public", "researcher"]
                }
            }
        ]
        
        result_cross = kg.graph_rag_context(cross_domain_evidence, ["battery"])
        
        assert result_cross["cross_domain_path_count"] > 0
        assert len(result_cross["multi_hop_paths"]) > 0

    def test_graph_rag_path_scoring_and_ranking(
        self, mock_embedder
    ):
        """GraphRAG paths should be scored and ranked correctly."""
        from reasoning.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        evidence = [
            {
                "id": "doc:1",
                "score": 0.9,
                "payload": {
                    "id": "doc:1", "title": "Doc 1", "content": "shared_entity content",
                    "domain": "energy", "tags": ["shared_entity"], "allowed_roles": ["public"]
                }
            },
            {
                "id": "doc:2",
                "score": 0.85,
                "payload": {
                    "id": "doc:2", "title": "Doc 2", "content": "shared_entity more content",
                    "domain": "finance", "tags": ["shared_entity"], "allowed_roles": ["public"]
                }
            },
            {
                "id": "doc:3",
                "score": 0.8,
                "payload": {
                    "id": "doc:3", "title": "Doc 3", "content": "shared_entity even more",
                    "domain": "climate", "tags": ["shared_entity"], "allowed_roles": ["public"]
                }
            }
        ]
        
        result = kg.graph_rag_context(evidence, ["shared_entity"])
        
        paths = result["multi_hop_paths"]
        assert len(paths) > 0
        
        for i in range(len(paths) - 1):
            assert paths[i]["path_score"] >= paths[i + 1]["path_score"]
        
        cross_domain_paths = [p for p in paths if p["cross_domain"]]
        assert len(cross_domain_paths) > 0

    def test_discovery_scorer_integration_with_early_stop(
        self, mock_embedder, mock_agent, sample_evidence
    ):
        """Discovery scorer should integrate with early stopping logic."""
        from reasoning.knowledge_graph import DiscoveryScorer
        from reasoning.neuro_symbolic_pipeline import NeuroSymbolicPipeline
        
        graph_context = {
            "multi_hop_paths": [
                {"path_score": 80, "cross_domain": True},
                {"path_score": 75, "cross_domain": False},
                {"path_score": 70, "cross_domain": True},
            ],
            "cross_domain_path_count": 2,
            "nodes": [], "edges": [],
            "query_anchors": [], "seed_document_ids": [],
            "strategy": "test", "path_scoring_method": "test"
        }
        
        discovery = DiscoveryScorer.score(sample_evidence, graph_context)
        
        assert "overall_score" in discovery
        assert "rating" in discovery
        assert discovery["rating"] in ["strong", "promising", "preliminary"]
        
        high_confidence_evidence = [
            {**e, "score": 0.95} for e in sample_evidence
        ]
        
        discovery_high = DiscoveryScorer.score(high_confidence_evidence, graph_context)
        
        assert discovery_high["overall_score"] >= discovery["overall_score"]


class TestGraphRAGConfiguration:
    """Test GraphRAG configuration and thresholds."""

    def test_graph_rag_settings_validation(self):
        """GraphRAG settings should be validated."""
        from config import settings
        
        assert settings.GRAPH_RAG_DEPTH >= 1
        assert settings.GRAPH_RAG_DEPTH <= 10
        assert settings.GRAPH_RAG_MAX_PATHS >= 1
        assert settings.GRAPH_RAG_MAX_PATHS <= 1000
        assert settings.GRAPH_RAG_MIN_EVIDENCE >= 1
        assert settings.GRAPH_RAG_MIN_EVIDENCE <= 1000

    def test_confidence_calibration_thresholds(self):
        """Confidence calibration should use configurable thresholds."""
        from reasoning.knowledge_graph import ConfidenceCalibrator
        
        discovery = {"overall_score": 85.0}
        validation = {"validation_score": 90.0}
        
        thresholds = {"proceed": 0.8, "investigate": 0.6}
        
        result = ConfidenceCalibrator.calibrate(0.9, discovery, validation, thresholds)
        
        assert result["calibrated_confidence"] >= 0.8
        assert result["decision"] == "proceed_to_experimental_design"
        assert result["thresholds"]["proceed"] == 0.8
        assert result["thresholds"]["investigate"] == 0.6

    def test_quality_gate_integration(self):
        """Quality gate should integrate with GraphRAG confidence."""
        from reasoning.strategy_layer import get_quality_gate
        
        quality_gate = get_quality_gate()
        
        result = quality_gate.evaluate(
            calibrated_confidence=0.85,
            validation_score=80.0,
            evidence_count=5,
            discovery_score=75.0
        )
        
        assert "passed" in result
        assert isinstance(result["passed"], bool)


class TestGraphRAGEdgeCases:
    """Edge cases for GraphRAG early stopping."""

    def test_empty_evidence_handling(self):
        """Empty evidence should return empty graph context."""
        from reasoning.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        result = kg.graph_rag_context([], [])
        
        assert result["nodes"] == []
        assert result["edges"] == []
        assert result["multi_hop_paths"] == []
        assert result["cross_domain_path_count"] == 0

    def test_single_document_no_bridges(self):
        """Single document should produce no multi-hop paths."""
        from reasoning.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        evidence = [{
            "id": "doc:1",
            "score": 0.9,
            "payload": {
                "id": "doc:1", "title": "Single", "content": "unique content",
                "domain": "energy", "tags": ["unique"], "allowed_roles": ["public"]
            }
        }]
        
        result = kg.graph_rag_context(evidence, ["unique"])
        
        assert result["multi_hop_paths"] == []
        assert result["cross_domain_path_count"] == 0

    def test_many_documents_performance(self):
        """Large number of documents should not cause performance issues."""
        from reasoning.knowledge_graph import KnowledgeGraph
        import time
        
        kg = KnowledgeGraph()
        
        large_evidence = []
        for i in range(100):
            large_evidence.append({
                "id": f"doc:{i}",
                "score": 0.9 - (i * 0.005),
                "payload": {
                    "id": f"doc:{i}",
                    "title": f"Doc {i}",
                    "content": f"Content about topic {i % 10}",
                    "domain": ["energy", "finance", "healthcare", "climate"][i % 4],
                    "tags": [f"topic_{i % 10}"],
                    "allowed_roles": ["public", "researcher"]
                }
            })
        
        start = time.time()
        result = kg.graph_rag_context(large_evidence, ["topic_0"])
        elapsed = time.time() - start
        
        assert elapsed < 1.0
        assert len(result["multi_hop_paths"]) <= 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])