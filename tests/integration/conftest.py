import os
import sys
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("QDRANT_IN_MEMORY", "true")
os.environ.setdefault("NEO4J_ENABLED", "false")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")

from config import settings
settings.ENVIRONMENT = "testing"
settings.QDRANT_IN_MEMORY = True
settings.NEO4J_ENABLED = False
settings.MULTI_AGENT_ENABLED = True
settings.Z3_VALIDATION_ENABLED = True
settings.DUAL_MEMORY_ENABLED = True
settings.EXPERIMENTAL_BLUEPRINT_ENABLED = True
settings.RISK_FEEDBACK_ENABLED = True
settings.LITELLM_ENABLED = True
settings.ZAYA1B_ENABLED = True
settings.ZAYA1_8B_REASONING_ENABLED = True
settings.GRAPH_RAG_ENABLED = True
settings.SCALLOP_ENABLED = False
settings.DEFORESTVIS_ENABLED = False
settings.SEMARA_ENABLED = True
settings.CROSS_ENCODER_RERANKING_ENABLED = True
settings.SEMANTIC_QUERY_CACHE_THRESHOLD = 0.92
settings.CACHE_SCHEMA_VERSION = 1
settings.DATA_SCHEMA_VERSION = 1
settings.VECTOR_SCHEMA_VERSION = 1
settings.KG_SCHEMA_VERSION = 1
settings.API_SCHEMA_VERSION = 1


@pytest.fixture(autouse=True)
def reset_singletons():
    import reasoning.neuro_symbolic_pipeline as nsp
    import ingestion.pipeline as ip
    import ingestion.ingestion_cache as ic
    import ingestion.redis_cache as rc
    import reasoning.query_cache as qc
    import reasoning.agent_registry as ar
    import reasoning.multi_agent as ma
    import vector_store.qdrant_engine as qe
    import ingestion.embedding as emb
    import reasoning.knowledge_graph as kg
    import reasoning.neo4j_graph as ng
    import reasoning.memory_service as ms
    import reasoning.dual_memory as dm
    import ingestion.queue_manager as qm
    import ingestion.dynamic_connectors as dc
    import reasoning.routing_metrics as rm
    import reasoning.strategy_layer as sl
    import reasoning.rule_updater as ru
    import reasoning.retrainer as rt
    import reasoning.benchmark_collector as bc
    import reasoning.feedback_collector as fc
    import reasoning.rule_engine as re
    import reasoning.hybrid_rag_kg as hrk
    import reasoning.evidence_attribution as ea
    import reasoning.risk_feedback as rf
    import reasoning.collaboration_recommender as cr
    import reasoning.query_classifier as qcl
    import reasoning.scallop as sc
    import reasoning.deforest_vis as dv
    import reasoning.wfa_fast_path as wf
    import reasoning.semara_reasoner as sr
    import reasoning.abductive_engine as ae
    import reasoning.experimental_blueprint as eb
    import reasoning.traceability as tr
    import reasoning.symbolic_filter as sf
    import reasoning.rxg_nano_agent as rna
    import reasoning.decision_tree as dt
    import reasoning.datalog_engine as de
    import reasoning.opa_enforcer as oe
    import reasoning.auth_service as auth
    import reasoning.user_profiles as up
    import reasoning.auto_discover as ad
    import reasoning.simulation_client as sim

    nsp._neuro_symbolic_pipeline = None
    ip._pipeline_instance = None
    ic._cache_instance = None
    rc._cache_instance = None
    qc._query_cache_instance = None
    ar._global_registry = None
    ma._orchestrator_instance = None
    qe._qdrant_engine_instance = None
    emb._embedder_instance = None
    kg._knowledge_graph = kg.KnowledgeGraph()
    ms._memory_service = None
    dm._dual_memory = None
    qm._queue_manager_instance = None
    rm._routing_metrics = None
    sl._unified_router = None
    sl._quality_gate = None
    sl._cost_controller = None
    ru._rule_updater = None
    rt._model_retrainer = None
    bc._benchmark_collector = None
    fc._feedback_collector = None
    re._rule_engine = None
    hrk._hybrid_rag_kg = None
    ea._evidence_attributor = None
    rf._risk_feedback_engine = None
    cr._collaboration_recommender = None
    qcl._query_classifier = None
    wf._wfa_engine = None
    sr._semara_reasoner = None
    ae._abductive_engine = None
    eb._experimental_blueprint_generator = None
    tr._evidence_traces = None
    sf._pre_filter = None
    sf._post_validator = None
    rna._agent = None
    auth._auth_service = None
    up._user_profile_service = None
    ad._auto_discover_engine = None
    sim._simulation_client = None

    yield


@pytest.fixture
def mock_qdrant_client():
    with patch("vector_store.qdrant_engine.QdrantClient") as mock:
        mock_client = Mock()
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client.upsert.return_value = None
        mock_client.search.return_value = []
        mock_client.query_points.return_value = Mock(points=[])
        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_redis_client():
    with patch("ingestion.redis_cache.redis") as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = None
        mock_client.setex.return_value = True
        mock_client.delete.return_value = True
        mock_client.flushdb.return_value = True
        mock_redis.Redis.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_neo4j_driver():
    with patch("reasoning.neo4j_graph.GraphDatabase") as mock_neo4j:
        mock_driver = Mock()
        mock_session = Mock()
        mock_session.run.return_value = []
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_neo4j.driver.return_value = mock_driver
        yield mock_driver


@pytest.fixture
def mock_embedder():
    with patch("ingestion.embedding.get_embedder") as mock:
        mock_embedder = Mock()
        mock_embedder.embed_text.return_value = [0.1] * 1024
        mock_embedder.embed_texts.return_value = [[0.1] * 1024]
        mock_embedder.normalize_vector.return_value = {"flat_vector": [0.1] * 256, "vector_meta": {"type": "dense"}}
        mock_embedder.reshape_vector.return_value = [0.1] * 256
        mock.return_value = mock_embedder
        yield mock_embedder


@pytest.fixture
def mock_agent():
    with patch("reasoning.rxg_nano_agent.ZAYA1_8BAgent") as mock:
        mock_agent = Mock()
        mock_agent.reason_and_synthesize.return_value = {
            "model": "ZAYA1-8B",
            "think_block": "Test reasoning",
            "tool_calls": [],
            "output_text": "Test hypothesis",
            "hypothesis": "Test hypothesis",
            "cited_evidence_ids": ["doc:1"],
            "confidence_score": 0.85,
        }
        mock_agent.stream_reasoning.return_value = iter([
            {"stage": "thinking", "delta": "Thinking..."},
            {"stage": "hypothesis_synthesis", "delta": "Hypothesis", "structured_result": {
                "model": "ZAYA1-8B",
                "think_block": "Test reasoning",
                "tool_calls": [],
                "output_text": "Test hypothesis",
                "hypothesis": "Test hypothesis",
                "cited_evidence_ids": ["doc:1"],
                "confidence_score": 0.85,
            }}
        ])
        mock_agent.model_name = "ZAYA1-8B"
        mock.return_value = mock_agent
        yield mock_agent


@pytest.fixture
def sample_documents() -> List[Dict[str, Any]]:
    return [
        {
            "id": "doc:1",
            "title": "Energy Storage in Battery Systems",
            "content": "Lithium-ion batteries are crucial for renewable energy storage. The electrolyte composition affects performance.",
            "domain": "energy",
            "tags": ["battery", "electrolyte", "renewable"],
            "allowed_roles": ["public", "researcher"],
        },
        {
            "id": "doc:2",
            "title": "Financial Market Risk Analysis",
            "content": "Portfolio risk assessment requires understanding volatility and correlation between assets.",
            "domain": "finance",
            "tags": ["portfolio", "risk", "volatility"],
            "allowed_roles": ["public", "researcher"],
        },
        {
            "id": "doc:3",
            "title": "Nanomaterial Drug Delivery",
            "content": "Lipid nanoparticles enable targeted drug delivery across the blood-brain barrier for neurological treatments.",
            "domain": "pharmacology",
            "tags": ["nanoparticle", "drug delivery", "blood-brain barrier"],
            "allowed_roles": ["public", "researcher"],
        },
    ]


@pytest.fixture
def sample_evidence() -> List[Dict[str, Any]]:
    return [
        {
            "id": "doc:1",
            "score": 0.92,
            "payload": {
                "id": "doc:1",
                "title": "Energy Storage in Battery Systems",
                "content": "Lithium-ion batteries are crucial for renewable energy storage.",
                "domain": "energy",
                "tags": ["battery", "electrolyte"],
                "allowed_roles": ["public", "researcher"],
            }
        },
        {
            "id": "doc:2",
            "score": 0.88,
            "payload": {
                "id": "doc:2",
                "title": "Financial Market Risk Analysis",
                "content": "Portfolio risk assessment requires understanding volatility.",
                "domain": "finance",
                "tags": ["portfolio", "risk"],
                "allowed_roles": ["public", "researcher"],
            }
        },
    ]


@pytest.fixture
def test_client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


@pytest.fixture
def auth_headers(test_client):
    if settings.API_KEY:
        return {"Authorization": f"Bearer {settings.effective_api_key}"}
    return {}