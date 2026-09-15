# CrossMind Operations & Migration Guide

## 1. Current Operational State

### Implemented Services

| Service | Implementation | Operational Notes |
|---------|---------------|-------------------|
| **FastAPI API Server** | `app/main.py` (343 lines) | Full FastAPI app with `/api/query`, `/api/ingest`, `/api/stream_reasoning`, `/v1/api/*` aliases, auth, versioning, validation, streaming, metrics, health. Exports `app` object. |
| **Reasoning Pipeline** | `reasoning/neuro_symbolic_pipeline.py` | Runnable directly via `NeuroSymbolicPipeline()` or through the API. Works end-to-end in tests (147 passed). |
| **Ingestion Pipeline** | `ingestion/pipeline.py` | Runnable directly. Supports batch and continuous ingestion. |
| **Vector Store** | `vector_store/qdrant_engine.py` | Qdrant client with in-memory fallback. No external Qdrant needed for local dev. |
| **Redis Cache** | `ingestion/redis_cache.py` | Optional. Graceful degradation when Redis is unavailable. |
| **Streamlit Dashboard** | `dashboard/app.py` | Runnable standalone. Connects to API base URL (configurable). |
| **Observability** | `app/observability.py` | Prometheus metrics + OpenTelemetry with graceful fallback when libraries are missing. |
| **Monitoring Stack** | `monitoring/` | Prometheus + Grafana configs are valid and ready for container deployment. |
| **Migrations** | `migrations/` | 3 schema/Redis/vector index versions with Redis-backed version tracking runner. |
| **Kubernetes** | `kubernetes/`, `kubernetes/base/`, `helm/crossmind/` | Base manifests deploy via kustomize; Helm chart for production. |
| **Validation Scripts** | `scripts/` | YAML validation, security scan, compile checks all pass. |

### Structural Packages (No Implementation)

| Package | Status | Impact |
|---------|--------|--------|
| **Core / Infra / Services** | **Structural** | Directories contain only `__init__.py`. Their intended responsibilities are currently fulfilled by `reasoning/`, `ingestion/`, and `vector_store/`. |
| **React Frontend** | **Optional** | Code exists in `frontend/src/` but is disabled by default (`REACT_UI_ENABLED=False`). Not wired to any running server. |
| **Neo4j Knowledge Graph** | **Optional** | Code exists (`reasoning/neo4j_graph.py`, `hybrid_rag_kg.py`) but `NEO4J_ENABLED=False` by default. Falls back to in-memory graph context. |
| **vLLM Model Serving** | **Optional** | Docker Compose defines a vLLM service, but `VLLM_ENABLED=False` by default. Requires GPU. |
| **Celery Async Ingestion** | **Optional** | Code references Celery in config, but `ASYNC_INGESTION_ENABLED=False` by default. |

## 2. Migration Guidance

### From Legacy to Current Structure

The project README references a `reasoning/` flat module layout. The current intent is a four-layer architecture:

```
app/      -> Presentation & API
core/     -> Orchestration core factories
infra/    -> Storage, cache, retrieval adapters
services/ -> Service entry points & orchestration helpers
```

**Current reality**: `core/`, `infra/`, and `services/` are empty. All implementation lives in:

- `reasoning/` (orchestration, reasoning, agents, validation, memory)
- `ingestion/` (document processing, embeddings, caching)
- `vector_store/` (Qdrant, BM25, vector adapters)

**Migration path** (if populating empty packages):

1. **Move orchestration factories** from `reasoning/neuro_symbolic_pipeline.py` into `core/pipeline.py`
2. **Move infrastructure adapters** from `vector_store/` and `ingestion/redis_cache.py` into `infra/`
3. **Move service entry points** from `ingestion/pipeline.py` into `services/ingestion.py`
4. **Keep `reasoning/`** for pure reasoning logic (agents, validators, classifiers)
5. **Populate `app/main.py`** with the actual FastAPI router definitions (already implemented — `app/main.py` is a real 343-line FastAPI app)

> **Warning**: Do not perform this migration without updating all import paths across 50+ modules and 147 tests.

### Database Migration

| From | To | Notes |
|------|----|-------|
| In-memory Qdrant | Persistent Qdrant | Set `QDRANT_IN_MEMORY=False` and mount `qdrant_storage` volume |
| No Redis | Redis 7+ | Set `REDIS_HOST` and ensure `redis_data` volume is mounted |
| No Neo4j | Neo4j 5.18+ | Set `NEO4J_ENABLED=True` and mount `neo4j_data` volume |
| DiskCache (local) | Shared filesystem | Set `DISK_CACHE_PATH` to a shared mount for multi-replica deployments |

### Schema Versioning

The project uses schema version flags in `config.py`:

```python
CACHE_SCHEMA_VERSION: int = 1
DATA_SCHEMA_VERSION: int = 1
VECTOR_SCHEMA_VERSION: int = 1
KG_SCHEMA_VERSION: int = 1
API_SCHEMA_VERSION: int = 1
```

**Migration rule**: Increment the relevant version when changing payload structures. The test suite (`test_cache_version_invalidation.py`) validates version-aware cache invalidation.

## 3. Operational Runbooks

### Starting the Dashboard

```bash
# Default: connects to http://localhost:8000
python -m streamlit run dashboard/app.py --server.port 8501

# Custom API base
set API_BASE=http://my-api-server:8000
python -m streamlit run dashboard/app.py --server.port 8501
```

### Running the Pipeline Standalone

```python
from reasoning.neuro_symbolic_pipeline import NeuroSymbolicPipeline

pipeline = NeuroSymbolicPipeline()
result = pipeline.process_query(
    query="Cross-domain links between Alzheimer biomarkers and nanomaterials",
    user_role="researcher",
    session_id="ops-demo"
)

# Key fields in result:
# - pre_filter: domains, entities, language
# - retrieved_evidence: fused retrieval results
# - agent_reasoning: hypothesis, confidence, model
# - post_validation: validation score, passed boolean
# - z3_formal_validation: Z3 execution mode and score
# - performance_metrics: timing, chunk counts, graph node counts
```

### Ingestion Operations

```python
from ingestion.pipeline import IngestionPipeline

pipeline = IngestionPipeline()
pipeline.auto_init()  # Starts connectors and continuous ingestion

# Batch ingest
doc_ids = pipeline.ingest_documents([
    {
        "title": "Document Title",
        "content": "Full text content...",
        "domain": "neuroscience",
        "tags": ["alzheimer", "biomarker"],
        "allowed_roles": ["public", "researcher"]
    }
])
```

### Continuous Ingestion

```python
from ingestion.continuous_ingestion import ContinuousIngestionWorker

worker = ContinuousIngestionWorker(pipeline=pipeline)
worker.start()  # Background daemon thread
# Check status
print(worker.get_status())
worker.stop()
```

### Monitoring Operations

#### Prometheus Metrics

The following metrics are exposed when the API server is running:

| Metric | Type | Labels |
|--------|------|--------|
| `crossmind_http_requests_total` | Counter | method, path, status |
| `crossmind_http_request_duration_seconds` | Histogram | method, path |
| `crossmind_queries_total` | Counter | decision |

#### Grafana Dashboards

Access Grafana at `http://localhost:3000` (when using Docker Compose). The pre-provisioned `CrossMind Overview` dashboard shows:

- API request rate by endpoint
- p95 API latency by endpoint
- Query decision distribution (fast/medium/deep)

#### Jaeger Tracing

Access Jaeger UI at `http://localhost:16686`. Traces are exported via OTLP to `http://jaeger:4317` when `OPENTELEMETRY_ENABLED=True`.

## 4. Scaling Considerations

### Horizontal Scaling

- **API pods**: `kubernetes/autoscaling.yaml` defines HPA with 2-10 replicas based on CPU (70%) and memory (75%).
- **vLLM pods**: HPA with 1-4 replicas based on request queue size and GPU utilization (80%).

### Cache Scaling

- **Redis**: Use Redis Cluster for >50k QPS. Current config uses single-node Redis.
- **Query cache**: In-memory LRU with configurable TTL (`REDIS_QUERY_CACHE_TTL`). For distributed deployments, replace with Redis-backed cache.

### Vector Store Scaling

- **Qdrant**: Use distributed Qdrant cluster for >100M vectors. Current config uses single-node with HNSW indexing.
- **Matryoshka embeddings**: Support dimensionality reduction (e.g., 1024 -> 256) without re-embedding.

## 5. Backup & Recovery

| Component | Backup Strategy | Recovery |
|-----------|----------------|----------|
| Qdrant | Volume snapshot (`qdrant_storage`) | Restore volume, restart container |
| Redis | RDB/AOF (`redis_data` volume) | Restore volume, restart container |
| Neo4j | Volume snapshot (`neo4j_data`) | Restore volume, restart container |
| DLDB | `DLDB_BACKUP_ENABLED` flag | Periodic backup to configured path |
| DiskCache | Filesystem backup of `DISK_CACHE_PATH` | Restore directory |

## 6. Troubleshooting

### "Agent with ID already registered"

Symptom: `ValueError: Agent with ID critic_agent already registered`

Cause: Singleton registry not reset between test runs or process restarts.

Fix:
```python
from reasoning.agent_registry import get_agent_registry
get_agent_registry().unregister("critic_agent")
```

### "Qdrant connection refused"

Symptom: Qdrant client fails to connect.

Fix: Ensure `QDRANT_IN_MEMORY=True` for local development, or start the Qdrant service:
```bash
docker compose up qdrant -d
```

### "Redis connection refused"

Symptom: Redis cache unavailable.

Fix: The system falls back to in-memory cache automatically. To use Redis:
```bash
docker compose up redis -d
```

### "No module named 'app.main'"

Symptom: Import errors in tests or health check.

Cause: This should not occur — `app/main.py` exists and exports a real FastAPI `app` object. If you encounter this, verify the Python path includes the repo root.

### High Memory Usage

Symptom: Process exceeds available RAM.

Mitigations:
1. Set `QDRANT_IN_MEMORY=False` and use external Qdrant
2. Reduce `CHUNK_SIZE` (default 512 tokens)
3. Disable `DUAL_MEMORY_ENABLED` if session memory is not needed
4. Reduce `REDIS_QUERY_CACHE_MAX` (default 5000)
