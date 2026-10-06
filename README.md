# CrossMind

CrossMind is a working neuro-symbolic scientific discovery stack built around a FastAPI API, a multi-agent reasoning engine, a retrieval layer, and a Streamlit dashboard. The project is structured to support real document ingestion, hybrid retrieval, evidence-aware reasoning, and operational monitoring without requiring a full enterprise deployment to be present at development time.

## What is included

The current implementation includes the following verified runtime paths:

- API and app server: [app/main.py](app/main.py)
- Ingestion and document processing: [ingestion](ingestion)
- Multi-agent orchestration: [reasoning/multi_agent.py](reasoning/multi_agent.py)
- Strategy, budget, and quality routing: [reasoning/strategy_layer.py](reasoning/strategy_layer.py)
- End-to-end reasoning pipeline: [reasoning/neuro_symbolic_pipeline.py](reasoning/neuro_symbolic_pipeline.py)
- Vector retrieval and storage adapters: [vector_store](vector_store)
- Deployment config and environment settings: [config.py](config.py)
- Docker service setup: [docker-compose.yml](docker-compose.yml)
- Dashboard UI: [dashboard/app.py](dashboard/app.py)
- Kubernetes manifests: [kubernetes/](kubernetes/) and [kubernetes/base/](kubernetes/base/)
- Production Helm charts: [helm/crossmind/](helm/crossmind/)
- Database migrations: [migrations/](migrations/)
- Validation and security scripts: [scripts/](scripts/)

## Runtime structure

The active runtime is organized into a clearer production layout:

- app: API entry points and presentation-facing application logic
- core: orchestration and reasoning core factories (structural package, implementation in reasoning/)
- infra: storage, cache, and retrieval infrastructure adapters (structural package, implementations in vector_store/ and ingestion/)
- services: higher-level service entry points and orchestration helpers (structural package, implementations in ingestion/ and reasoning/)

This structure is represented by the packages:

- [app](app)
- [core](core)
- [infra](infra)
- [services](services)

## Verified stack

### Ingestion and retrieval

- Document extraction and chunking utilities in [ingestion](ingestion)
- Qdrant-backed vector storage in [vector_store/qdrant_engine.py](vector_store/qdrant_engine.py)
- Embedding and vector normalization in [ingestion/embedding.py](ingestion/embedding.py) and [vector_store/vector_adapter.py](vector_store/vector_adapter.py)
- Redis-backed cache and fallback behavior in [ingestion/redis_cache.py](ingestion/redis_cache.py)
- Knowledge graph support and graph context generation in [reasoning/knowledge_graph.py](reasoning/knowledge_graph.py)

### Reasoning and orchestration

- Specialist agents, critic, synthesizer, and orchestrator in [reasoning/multi_agent.py](reasoning/multi_agent.py)
- Unified routing and budget logic in [reasoning/strategy_layer.py](reasoning/strategy_layer.py)
- Full reasoning pipeline in [reasoning/neuro_symbolic_pipeline.py](reasoning/neuro_symbolic_pipeline.py)
- Query classification and routing in [reasoning/query_classifier.py](reasoning/query_classifier.py)
- Memory and session context in [reasoning/dual_memory.py](reasoning/dual_memory.py) and [reasoning/memory_service.py](reasoning/memory_service.py)

### Application layer

- FastAPI API endpoints, auth, validation, and streaming in [app/main.py](app/main.py)
- Streamlit dashboard in [dashboard/app.py](dashboard/app.py)
- Service configuration, feature flags, and environment defaults in [config.py](config.py)

## Environment and deployment

The project supports local and containerized execution with Docker Compose. The stack includes:

- Redis
- Qdrant
- Neo4j
- Jaeger
- Prometheus
- Grafana
- optional vLLM service for model serving

See [docker-compose.yml](docker-compose.yml) for the live deployment setup.

## Quick start

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run the API locally

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

3. Run the dashboard locally

```bash
python -m streamlit run dashboard/app.py --server.port 8501
```

4. Optional: run the full Docker stack

```bash
docker compose up -d
```

5. Optional: deploy to Kubernetes via Helm

```bash
helm install crossmind helm/crossmind
```

The FastAPI app in [app/main.py](app/main.py) exports the canonical endpoints `/api/query`, `/api/public-research/query`, `/api/ingest`, `/api/stream_reasoning`, `/healthz`, `/metrics`, `/auth/login`, `/auth/refresh`, `/auth/users`, and `/`. The redundant `/v1/api/*` aliases were removed. All endpoints support Bearer-token authentication except health and metrics. The API includes request validation via Pydantic schemas, Server-Sent Events streaming, Prometheus metrics, and OpenAPI documentation at `/docs`.

### Public research API

`POST /api/public-research/query` accepts a validated query, optional `max_results` (1-20), `user_role`, and `session_id`. The typed sequence is:

1. QueryRouter selects a public provider and normalized keyword terms.
2. The provider adapter searches Europe PMC, PubMed, OpenAlex, arXiv, or Semantic Scholar without requiring an API key.
3. Normalized records are ingested into the document pipeline.
4. The retrieved evidence and graph context enter the neuro-symbolic pipeline.
5. UnifiedRouter always selects `execution_mode: deep`, hybrid RAG-KG retrieval, three agents, graph RAG, and a 6000-token budget.

Provider failures are isolated and logged; successful results are preserved. Generic document ingestion remains available through `/api/ingest`, and document records may contain arbitrary public or researcher-visible metadata while enforcing explicit content and size limits.

## Important runtime notes

- The project is structured and tested as a working runtime stack, not a purely theoretical architecture.
- The system includes compatibility fallback behavior so that a local startup can proceed even when some external services are not available.
- The multi-agent pipeline is implemented and validated in the current workspace.
- The main operational path is the FastAPI app plus the reasoning pipeline; the extra experimental and archival modules were intentionally moved out of the active root to keep the repo clean and maintainable.

## Validation status

The focused public research suite passes all 7 tests. The full suite is currently blocked by existing persistent-Qdrant test configuration: tests that patch the client still initialize Qdrant with `QDRANT_IN_MEMORY=True`, which the production workflow rejects. Run the repository test command after configuring a persistent Qdrant service or the test environment's in-memory override.

```bash
python -m pytest tests
```

Packaging and validation commands:

```bash
python -c "import tomli; tomli.load(open('pyproject.toml','rb'))"
python scripts/validate_yaml.py .github/workflows monitoring kubernetes helm
python scripts/security_check.py --skip-pip-audit
python -m compileall app ingestion reasoning vector_store dashboard scripts migrations
```

## License

Apache 2.0