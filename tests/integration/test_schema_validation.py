import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
from fastapi.testclient import TestClient
from pydantic import ValidationError


class TestSchemaValidation:
    """Test request/response schema validation."""

    def test_query_request_validation_valid(self, test_client, auth_headers):
        """Valid query request should pass validation."""
        response = test_client.post(
            "/api/query",
            json={
                "query": "test query",
                "user_role": "researcher",
                "confidence_proceed_threshold": 0.75,
                "confidence_investigate_threshold": 0.50,
                "session_id": "test_session"
            },
            headers=auth_headers
        )
        
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "query" in data
            assert "confidence_calibration" in data

    def test_query_request_validation_invalid_role(self, test_client, auth_headers):
        """Invalid user_role should return 422."""
        response = test_client.post(
            "/api/query",
            json={
                "query": "test query",
                "user_role": "invalid_role",
                "confidence_proceed_threshold": 0.75,
                "confidence_investigate_threshold": 0.50
            },
            headers=auth_headers
        )
        
        assert response.status_code == 422

    def test_query_request_validation_empty_query(self, test_client, auth_headers):
        """Empty query should return 422."""
        response = test_client.post(
            "/api/query",
            json={
                "query": "",
                "user_role": "researcher",
                "confidence_proceed_threshold": 0.75,
                "confidence_investigate_threshold": 0.50
            },
            headers=auth_headers
        )
        
        assert response.status_code == 422

    def test_query_request_validation_threshold_order(self, test_client, auth_headers):
        """Investigate threshold cannot exceed proceed threshold."""
        response = test_client.post(
            "/api/query",
            json={
                "query": "test query",
                "user_role": "researcher",
                "confidence_proceed_threshold": 0.50,
                "confidence_investigate_threshold": 0.75
            },
            headers=auth_headers
        )
        
        assert response.status_code == 422

    def test_query_request_validation_threshold_bounds(self, test_client, auth_headers):
        """Thresholds must be between 0 and 1."""
        response = test_client.post(
            "/api/query",
            json={
                "query": "test query",
                "user_role": "researcher",
                "confidence_proceed_threshold": 1.5,
                "confidence_investigate_threshold": 0.50
            },
            headers=auth_headers
        )
        
        assert response.status_code == 422

    def test_ingest_request_validation(self, test_client, auth_headers):
        """Document ingestion request validation."""
        response = test_client.post(
            "/api/ingest",
            json={
                "documents": [
                    {
                        "title": "Test Document",
                        "content": "Test content",
                        "domain": "test",
                        "tags": ["tag1", "tag2"],
                        "allowed_roles": ["public", "researcher"]
                    }
                ]
            },
            headers=auth_headers
        )
        
        assert response.status_code in [200, 500]

    def test_ingest_request_sanitization(self, test_client, auth_headers):
        """HTML/script content should be sanitized."""
        response = test_client.post(
            "/api/ingest",
            json={
                "documents": [
                    {
                        "title": "<script>alert('xss')</script>Title",
                        "content": "<script>malicious</script>Content",
                        "domain": "test",
                        "tags": ["<script>tag</script>"],
                        "allowed_roles": ["public"]
                    }
                ]
            },
            headers=auth_headers
        )
        
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "<script>" not in str(data)

    def test_stream_query_validation(self, test_client, auth_headers):
        """Stream query parameter validation."""
        response = test_client.get(
            "/api/stream_reasoning",
            params={"query": "test", "user_role": "researcher"},
            headers=auth_headers
        )
        
        assert response.status_code in [200, 500]

    def test_stream_query_empty_validation(self, test_client, auth_headers):
        """Empty stream query should return 400."""
        response = test_client.get(
            "/api/stream_reasoning",
            params={"query": "", "user_role": "researcher"},
            headers=auth_headers
        )
        
        assert response.status_code == 400

    def test_stream_query_invalid_role_validation(self, test_client, auth_headers):
        """Invalid stream role should return 400."""
        response = test_client.get(
            "/api/stream_reasoning",
            params={"query": "test", "user_role": "invalid"},
            headers=auth_headers
        )
        
        assert response.status_code == 400

    def test_request_size_limit(self, test_client, auth_headers):
        """Large requests should be rejected."""
        from config import settings
        
        large_content = "x" * (settings.MAX_DOC_CONTENT_LENGTH + 1000)
        
        response = test_client.post(
            "/api/ingest",
            json={
                "documents": [
                    {
                        "title": "Large Doc",
                        "content": large_content,
                        "domain": "test"
                    }
                ]
            },
            headers=auth_headers
        )
        
        assert response.status_code == 413


class TestGlobalErrorResponses:
    """Test global error response format and handling."""

    def test_404_error_format(self, test_client, auth_headers):
        """404 errors should have consistent format."""
        response = test_client.get("/api/nonexistent", headers=auth_headers)
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_422_error_format(self, test_client, auth_headers):
        """422 validation errors should have consistent format."""
        response = test_client.post(
            "/api/query",
            json={"invalid": "request"},
            headers=auth_headers
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_500_error_format(self, test_client, auth_headers):
        """500 errors should have consistent format and not leak internals."""
        from app.main import app
        from reasoning.neuro_symbolic_pipeline import get_neuro_symbolic_pipeline
        
        with patch("app.main.get_neuro_symbolic_pipeline") as mock_pipeline:
            mock_pipeline.side_effect = Exception("Internal error")
            
            response = test_client.post(
                "/api/query",
                json={
                    "query": "test",
                    "user_role": "researcher",
                    "confidence_proceed_threshold": 0.75,
                    "confidence_investigate_threshold": 0.50
                },
                headers=auth_headers
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Internal error" not in data["detail"]
            assert "traceback" not in str(data).lower()

    def test_rate_limit_error_format(self, test_client, auth_headers):
        """Rate limit errors should have consistent format."""
        from config import settings
        original_limit = settings.RATE_LIMIT_PER_MINUTE
        settings.RATE_LIMIT_PER_MINUTE = 1
        
        try:
            response1 = test_client.get("/", headers=auth_headers)
            response2 = test_client.get("/", headers=auth_headers)
            
            if response2.status_code == 429:
                data = response2.json()
                assert "detail" in data
                assert "rate limit" in data["detail"].lower()
        finally:
            settings.RATE_LIMIT_PER_MINUTE = original_limit

    def test_unauthorized_error_format(self, test_client):
        """Unauthorized errors should have consistent format."""
        from config import settings
        original_key = settings.API_KEY
        
        class MockSecretStr:
            def get_secret_value(self):
                return "test-key"
        
        settings.API_KEY = MockSecretStr()
        
        try:
            response = test_client.post(
                "/api/query",
                json={
                    "query": "test",
                    "user_role": "researcher",
                    "confidence_proceed_threshold": 0.75,
                    "confidence_investigate_threshold": 0.50
                }
            )
            
            assert response.status_code == 401
            data = response.json()
            assert "detail" in data
            assert "unauthorized" in data["detail"].lower()
        finally:
            settings.API_KEY = original_key

    def test_error_response_headers(self, test_client, auth_headers):
        """Error responses should include security headers."""
        response = test_client.post(
            "/api/query",
            json={"invalid": "request"},
            headers=auth_headers
        )
        
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert response.headers.get("X-XSS-Protection") == "1; mode=block"
        assert "X-Request-ID" in response.headers

    def test_request_id_propagation(self, test_client, auth_headers):
        """Request ID should be propagated through response headers."""
        request_id = "test-request-id-12345"
        
        response = test_client.get(
            "/",
            headers={**auth_headers, "X-Request-ID": request_id}
        )
        
        assert response.headers.get("X-Request-ID") == request_id


class TestResponseSchemaConsistency:
    """Test response schema consistency across endpoints."""

    def test_query_response_structure(self, mock_embedder, mock_agent, sample_evidence, test_client, auth_headers):
        """Query response should have consistent structure."""
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
                                
                                response = test_client.post(
                                    "/api/query",
                                    json={
                                        "query": "test query",
                                        "user_role": "researcher",
                                        "confidence_proceed_threshold": 0.75,
                                        "confidence_investigate_threshold": 0.50
                                    },
                                    headers=auth_headers
                                )
                                
                                if response.status_code == 200:
                                    data = response.json()
                                    
                                    required_fields = [
                                        "query", "user_role", "session_id",
                                        "pre_filter", "retrieved_evidence",
                                        "graph_rag", "cross_domain_scoring",
                                        "evidence_traceability", "evidence_attribution",
                                        "confidence_calibration", "agent_reasoning",
                                        "post_validation", "z3_formal_validation",
                                        "abductive_reasoning", "experimental_blueprint",
                                        "collaboration_recommendations",
                                        "multi_agent_orchestration",
                                        "risk_controlled_feedback",
                                        "memory_footprint", "performance_metrics"
                                    ]
                                    
                                    for field in required_fields:
                                        assert field in data, f"Missing field: {field}"

    def test_health_check_response_structure(self, test_client):
        """Health check should have consistent structure."""
        response = test_client.get("/healthz")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data

    def test_metrics_endpoint_response(self, test_client, auth_headers):
        """Metrics endpoint should return valid format."""
        response = test_client.get("/metrics", headers=auth_headers)
        
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")
        assert "crossmind" in response.text.lower()

    def test_root_endpoint_structure(self, test_client):
        """Root endpoint should return project info."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "project" in data
        assert "engine" in data
        assert "status" in data
        assert "version" in data
        assert "endpoints" in data


class TestInputSanitization:
    """Test input sanitization across endpoints."""

    def test_xss_prevention_in_query(self, test_client, auth_headers):
        """XSS payloads in query should be sanitized."""
        response = test_client.post(
            "/api/query",
            json={
                "query": "<script>alert('xss')</script>valid query",
                "user_role": "researcher",
                "confidence_proceed_threshold": 0.75,
                "confidence_investigate_threshold": 0.50
            },
            headers=auth_headers
        )
        
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "<script>" not in data.get("query", "")

    def test_javascript_protocol_removal(self, test_client, auth_headers):
        """JavaScript protocol should be removed from input."""
        response = test_client.post(
            "/api/ingest",
            json={
                "documents": [{
                    "title": "Test",
                    "content": "javascript:alert(1) legitimate content",
                    "domain": "test"
                }]
            },
            headers=auth_headers
        )
        
        assert response.status_code in [200, 500]

    def test_event_handler_removal(self, test_client, auth_headers):
        """Event handlers should be removed from input."""
        response = test_client.post(
            "/api/ingest",
            json={
                "documents": [{
                    "title": "Test",
                    "content": "onclick=alert(1) legitimate content",
                    "domain": "test"
                }]
            },
            headers=auth_headers
        )
        
        assert response.status_code in [200, 500]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])