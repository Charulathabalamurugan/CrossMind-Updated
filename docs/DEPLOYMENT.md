# CrossMind Deployment & Runtime Directory Map

## 1. Runtime Directory Map

The following map clarifies the intended production layout versus the current source tree.

```
E:\CrossMind-Updated/
│
├── app/                          # APPLICATION LAYER
│   ├── __init__.py
│   ├── main.py                   # FastAPI entrypoint (343 lines, real app)
│   ├── observability.py          # Prometheus metrics, OpenTelemetry, structured logging
│   └── schemas.py                # Pydantic v2 request/response validation
│
├── core/                         # CORE LAYER (planned, currently empty)
│   └── __init__.py
│   # Intended: orchestration factories, pipeline wiring, strategy initialization
│   # Current: responsibilities live in reasoning/ and ingestion/
│
├── infra/                        # INFRASTRUCTURE LAYER (planned, currently empty)
│   └── __init__.py
│   # Intended: storage, cache, retrieval infrastructure adapters
│   # Current: implementations live in vector_store/ and ingestion/
│
├── services/                     # SERVICE LAYER (planned, currently empty)
│   └── __init__.py
│   # Intended: higher-level service entry points and orchestration helpers
│   # Current: implementations live in ingestion/ and reasoning/
│
├── reasoning/                    # REASONING CORE (primary implementation)
│   ├── __init__.py
│   ├── neuro_symbolic_pipeline.py    # Main query orchestrator (685 lines, works end-to-end in tests)
│   ├── strategy_layer.py             # UnifiedRouter, QualityGate, CostController
│   ├── multi_agent.py                # Multi-agent orchestration, message bus
│   ├── agent_registry.py             # Agent lifecycle, capabilities, registry
│   ├── knowledge_graph.py            # GraphRAG context, discovery scoring
│   ├── hybrid_rag_kg.py              # Hybrid RAG + Knowledge Graph retrieval
│   ├── z3_validator.py               # Formal Z3 symbolic validation
│   ├── dual_memory.py                # Session + long-term memory
│   ├── memory_service.py             # Memory service singleton
│   ├── wfa_fast_path.py              # Weighted Finite Automata fast path
│   ├── decision_tree.py              # Decision tree classifier
│   ├── query_classifier.py           # Query complexity classification
│   ├── symbolic_filter.py            # Pre-filter and post-validation
│   ├── rxg_nano_agent.py             # ZAYA1-8B agent wrapper
│   ├── abductive_engine.py           # Abductive reasoning
│   ├── experimental_blueprint.py     # Experimental blueprint generation
│   ├── evidence_attribution.py       # Evidence attribution scoring
│   ├── risk_feedback.py              # Risk-controlled feedback
│   ├── collaboration_recommender.py  # Cross-domain collaboration recommendations
│   ├── query_cache.py                # Query result caching
│   ├── sparse_retriever.py           # BM25 sparse retrieval
│   ├── rule_engine.py                # Dynamic rule engine
│   ├── rule_updater.py               # Rule update mechanism
│   ├── retrainer.py                  # Model retraining controller
│   ├── feedback_collector.py         # User feedback collection
│   ├── benchmark_collector.py        # Benchmark tracking
│   ├── traceability.py               # Evidence traceability
│   ├── prometheus_monitor.py         # Prometheus monitoring integration
│   ├── dldb.py                       # Disk-backed lightweight database
│   ├── drift_detector.py             # Model drift detection
│   ├── plugin_manager.py             # Plugin discovery and loading
│   ├── auto_discover.py              # Autonomous hypothesis discovery
│   ├── simulation_client.py          # Simulation/fallback client
│   ├── opa_enforcer.py               # OPA policy enforcement
│   ├── auth_service.py               # Authentication service
│   ├── user_profiles.py              # User profile management
│   ├── semara_reasoner.py            # Semara symbolic reasoner
│   ├── scallop.py                    # Scallop neuro-symbolic reasoner
│   ├── deforest_vis.py               # Deforest visualization
│   ├── datalog_engine.py              # Datalog rule engine
│   ├── bridge_scorer.py              # Cross-domain bridge scoring
│   ├── result_formatter.py           # Pipeline result formatting
│   └── ...                           # Additional modules
│
├── ingestion/                    # INGESTION PIPELINE
│   ├── pipeline.py               # Unified 6-phase ingestion pipeline
│   ├── embedding.py              # BGE-M3 embedding with Matryoshka
│   ├── chunker.py                # Text chunking strategies
│   ├── text_extractor.py         # MinerU / Tika extraction
│   ├── mineru_extractor.py       # MinerU PDF extraction
│   ├── sparse_vector.py          # Sparse vector generation
│   ├── redis_cache.py            # Redis query/result cache
│   ├── ingestion_cache.py        # Ingestion-level cache
│   ├── queue_manager.py          # Async ingestion queue
│   ├── continuous_ingestion.py   # Background continuous worker
│   ├── active_learning.py        # Active learning engine
│   └── dynamic_connectors.py     # Dynamic connector framework
│
├── vector_store/                 # VECTOR RETRIEVAL
│   ├── qdrant_engine.py          # Qdrant client + BM25 + RBAC + fallback
│   └── vector_adapter.py         # Vector normalization, reshaping, multi-vector
│
├── dashboard/                    # STREAMLIT DASHBOARD (active UI)
│   ├── app.py                    # 633-line authenticated dashboard
│   └── __init__.py
│
├── frontend/                     # REACT FRONTEND (optional)
│   ├── package.json              # React 18 + dependencies
│   └── src/
│       ├── App.js                # Main React app
│       ├── Dashboard.js          # Dashboard router
│       ├── index.js              # Entry point
│       ├── index.css             # Styles
│       └── phases/
│           ├── Phase1Ingestion.js    # Async document ingestion UI
│           ├── Phase2Retrieval.js    # Hybrid retrieval UI
│           ├── Phase3Reasoning.js    # Neuro-symbolic reasoning UI
│           ├── Phase4Enrichment.js   # Enrichment & memory UI
│           ├── Phase5Streaming.js    # SSE streaming UI
│           └── Phase6Learning.js     # Continuous learning UI
│
├── monitoring/                   # OBSERVABILITY INFRASTRUCTURE
│   ├── prometheus.yml            # Prometheus scrape config
│   └── grafana/
│       ├── provisioning/
│       │   ├── dashboards/
│       │   │   ├── dashboard.yml          # Grafana dashboard provider
│       │   │   └── json/
│       │   │       └── crossmind-overview.json  # Overview dashboard
│       │   └── datasources/
│       │       └── prometheus.yml         # Prometheus datasource config
│       └── (volumes mounted at runtime)
│
├── kubernetes/                   # PRODUCTION K8s MANIFESTS
│   ├── autoscaling.yaml          # HPA for API (2-10 replicas) and vLLM (1-4)
│   └── ingress-https.yaml        # TLS ingress with nginx, cert-manager
│
├── tests/                        # TEST SUITE
│   ├── integration/              # Integration tests
│   │   ├── conftest.py           # Shared fixtures, singleton resets
│   │   ├── test_agent_registry.py
│   │   ├── test_cache_version_invalidation.py
│   │   ├── test_concurrent_requests.py
│   │   ├── test_graphrag_early_stopping.py
│   │   ├── test_plugin_discovery.py
│   │   ├── test_service_failure_degradation.py
│   │   ├── test_streaming.py
│   │   └── test_schema_validation.py
│   ├── test_feature_optimizations.py
│   ├── test_multi_agent_framework.py
│   └── test_runtime_compatibility.py
│
├── archive/                      # ARCHIVED LEGACY CODE
│   └── legacy/
│       ├── demo_query.py
│       ├── check_impl.py
│       └── experimental/
│
├── docker-compose.yml            # Full stack orchestration
├── Dockerfile                    # API service (Python 3.10-slim)
├── Dockerfile.dashboard          # Dashboard service (Python 3.10-slim)
├── config.py                     # Pydantic Settings (261 feature flags)
├── requirements.txt              # Python dependencies
├── health_check.py               # Component import validation script
├── run_api.bat                   # Windows launcher for API
├── run_streamlit.bat             # Windows launcher for dashboard
├── stream_demo.py                # Streaming demo script
├── live_test.py                  # Live integration test script
├── trace_filter.py               # Trace filtering utility
├── trace_search.py               # Trace search utility
├── check_pkgs.py                 # Package verification script
├── .env.example                  # Environment variable template
└── README.md                     # Project overview
```

## 2. Deployment Modes

### Local Development (Recommended)

```bash
# 1. Activate venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Streamlit dashboard (active UI)
python -m streamlit run dashboard/app.py --server.port 8501

# 4. In another terminal, test pipeline directly
python -c "from reasoning.neuro_symbolic_pipeline import NeuroSymbolicPipeline; \
           p = NeuroSymbolicPipeline(); \
           print(p.process_query('test query', user_role='researcher'))"
```

### Docker Compose (Infrastructure Stack)

```bash
docker compose up -d
```

**Services started**:

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `crossmind-redis` | `redis:7-alpine` | 6379 | Cache and queue |
| `crossmind-qdrant` | `qdrant/qdrant:v1.7.4` | 6333, 6334 | Vector storage |
| `crossmind-neo4j` | `neo4j:5.18` | 7474, 7687 | Knowledge graph |
| `crossmind-jaeger` | `jaegertracing/all-in-one:1.57` | 16686, 4317, 4318 | Distributed tracing |
| `crossmind-vllm` | `vllm/vllm-openai:latest` | 8001 | LLM serving (GPU required) |
| `crossmind-api` | Built from Dockerfile | 8000 | API service |
| `crossmind-dashboard` | Built from Dockerfile.dashboard | 8501 | Streamlit dashboard |
| `crossmind-prometheus` | `prom/prometheus:v2.53.1` | 9090 | Metrics collection |
| `crossmind-grafana` | `grafana/grafana:11.1.0` | 3000 | Metrics visualization |

> **Important**: The `crossmind-api` container builds from `Dockerfile` and runs `app/main.py`, which exports a real FastAPI `app` object exposed on port 8000.

### Kubernetes (Production)

Two manifest locations exist:

1. **Base manifests** in `kubernetes/base/` (deploy via kustomize):

```bash
kubectl kustomize kubernetes/base | kubectl apply -f -
```

2. **Helm chart** in `helm/crossmind/` (recommended for production):

```bash
helm install crossmind helm/crossmind
```

**Base manifests include**: namespace, configmap, secret-template, serviceaccount, deployment, service, ingress, HPA, PDB, networkpolicy.

**Helm chart includes**: Deployment, Service, Ingress, HPA, PDB, NetworkPolicy, ConfigMap, Secret, ServiceAccount, and helper templates.

Both target the same API service which exposes `/api/query`, `/api/ingest`, `/api/stream_reasoning`, `/v1/api/*`, `/healthz`, `/metrics`, and auth endpoints.

**Prerequisites**:
- Kubernetes cluster with GPU support (for vLLM)
- cert-manager for TLS
- nginx-ingress controller
- Secrets: `crossmind-tls` with base64-encoded certificate and key

## 3. Frontend & Monitoring Paths

### Frontend

| Path | Type | Status |
|------|------|--------|
| `dashboard/app.py` | Streamlit (Python) | **Active** - Default UI |
| `frontend/src/phases/Phase1Ingestion.js` | React | **Optional** - Disabled by default |
| `frontend/src/phases/Phase2Retrieval.js` | React | **Optional** - Disabled by default |
| `frontend/src/phases/Phase3Reasoning.js` | React | **Optional** - Disabled by default |
| `frontend/src/phases/Phase4Enrichment.js` | React | **Optional** - Disabled by default |
| `frontend/src/phases/Phase5Streaming.js` | React | **Optional** - Disabled by default |
| `frontend/src/phases/Phase6Learning.js` | React | **Optional** - Disabled by default |

### Monitoring

| Path | Type | Status |
|------|------|--------|
| `monitoring/prometheus.yml` | Prometheus config | **Implemented** |
| `monitoring/grafana/provisioning/datasources/prometheus.yml` | Grafana datasource | **Implemented** |
| `monitoring/grafana/provisioning/dashboards/dashboard.yml` | Grafana provider | **Implemented** |
| `monitoring/grafana/provisioning/dashboards/json/crossmind-overview.json` | Grafana dashboard | **Implemented** - Tracks API request rate, p95 latency, query decisions |

## 4. Environment Profiles

| Profile | ENVIRONMENT | Key Differences |
|---------|-------------|-----------------|
| `development` | `development` | All local fallbacks, in-memory Qdrant, verbose logging |
| `testing` | `testing` | Mocked external services, deterministic embeddings, singleton resets |
| `staging` | `staging` | External services expected, feature flags for gradual rollout |
| `production` | `production` | Strict validation, no fallbacks, external services required |

## 5. Health Check

Run the included health check script to validate component imports:

```bash
python health_check.py
```

Expected output shows OK/SKIPPED/ERROR for each phase component. All components should report OK or SKIPPED when dependencies are installed.
