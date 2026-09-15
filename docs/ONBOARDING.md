# CrossMind Onboarding Guide

## Prerequisites

- Python 3.10+
- pip
- Git
- (Optional) Docker Desktop with Compose
- (Optional) Node.js 18+ for React frontend

## 1. Clone and Setup

```bash
git clone <repository-url>
cd CrossMind-Updated
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 2. Environment Configuration

Copy `.env.example` to `.env` and review settings:

```bash
cp .env.example .env
```

Key variables to understand:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ENVIRONMENT` | `development` | Runtime profile |
| `QDRANT_IN_MEMORY` | `True` | Use in-memory Qdrant (no external service needed) |
| `REDIS_HOST` | `localhost` | Redis host |
| `NEO4J_ENABLED` | `False` | Enable Neo4j knowledge graph |
| `REACT_UI_ENABLED` | `False` | Enable React frontend |
| `VLLM_ENABLED` | `False` | Enable vLLM model serving |
| `USE_LOCAL_SIMULATOR_FALLBACK` | `True` | Fallback when LLM unavailable |
| `ZAYA1_8B_REASONING_ENABLED` | `True` | Enable ZAYA1-8B reasoning path |

## 3. Project Structure

```
E:\CrossMind-Updated/
├── app/                    # API entrypoints + observability + schemas
│   ├── main.py             # FastAPI entrypoint (343 lines, real app)
│   ├── observability.py    # Prometheus, OpenTelemetry, logging
│   └── schemas.py          # Pydantic request/response models
├── core/                   # Reserved for orchestration core (empty, __init__.py only)
├── infra/                  # Reserved for storage/cache adapters (empty, __init__.py only)
├── services/               # Reserved for service orchestration (empty, __init__.py only)
├── reasoning/              # **Primary implementation: 50+ modules**
│   ├── neuro_symbolic_pipeline.py  # Main query orchestrator (685 lines, works end-to-end)
│   ├── strategy_layer.py           # Unified router, quality gate, cost controller
│   ├── multi_agent.py              # Multi-agent orchestration
│   ├── agent_registry.py           # Agent lifecycle management
│   ├── knowledge_graph.py          # GraphRAG and discovery scoring
│   ├── hybrid_rag_kg.py            # Hybrid RAG + Knowledge Graph retrieval
│   ├── z3_validator.py             # Formal Z3 validation
│   ├── dual_memory.py              # Session + long-term memory
│   ├── wfa_fast_path.py            # Weighted Finite Automata fast path
│   ├── decision_tree.py            # Decision tree classifier
│   └── ...                         # Additional reasoning modules
├── ingestion/              # Document processing pipeline
│   ├── pipeline.py         # Unified ingestion pipeline (6 phases)
│   ├── embedding.py        # BGE-M3 embedding with Matryoshka support
│   ├── chunker.py          # Document chunking
│   ├── text_extractor.py   # Text extraction (MinerU, Tika)
│   ├── redis_cache.py      # Redis-backed caching
│   └── ...
├── vector_store/           # Vector retrieval
│   ├── qdrant_engine.py    # Qdrant client with BM25 + in-memory fallback
│   └── vector_adapter.py   # Vector normalization and reshaping
├── dashboard/              # Streamlit dashboard (active UI)
│   └── app.py              # 633-line authenticated dashboard
├── frontend/               # React frontend (optional, disabled by default)
│   └── src/phases/         # Phase1-6 React components
├── monitoring/             # Observability infrastructure
│   ├── prometheus.yml      # Prometheus scrape config
│   └── grafana/            # Grafana provisioning + dashboard JSON
├── kubernetes/             # Production Kubernetes manifests
│   ├── autoscaling.yaml    # HPA for API, vLLM, and Qdrant
│   ├── ingress-https.yaml  # TLS ingress
│   └── base/               # Base manifests (kustomize)
│       ├── namespace.yaml
│       ├── configmap.yaml
│       ├── secret-template.yaml
│       ├── serviceaccount.yaml
│       ├── deployment.yaml
│       ├── service.yaml
│       ├── ingress.yaml
│       ├── hpa.yaml
│       ├── pdb.yaml
│       ├── networkpolicy.yaml
│       └── kustomization.yaml
├── helm/crossmind/         # Production Helm chart
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/
│       ├── deployment.yaml
│       ├── service.yaml
│       ├── ingress.yaml
│       ├── hpa.yaml
│       ├── pdb.yaml
│       ├── networkpolicy.yaml
│       ├── configmap.yaml
│       ├── secret.yaml
│       ├── serviceaccount.yaml
│       └── _helpers.tpl
├── migrations/             # Database migrations with runner
│   ├── runner.py           # MigrationRunner with Redis version tracking
│   └── versions/
│       ├── v001_initial_schema.py
│       ├── v002_add_vector_indexes.py
│       └── v003_redis_keyspaces.py
├── scripts/                # Validation, security, and benchmark scripts
│   ├── validate_yaml.py
│   ├── security_check.py
│   └── benchmarking/
├── tests/                  # Test suite (147 passed)
│   ├── integration/        # Integration tests
│   └── test_*.py           # Unit tests
├── docker-compose.yml      # Full infrastructure stack
├── Dockerfile              # API service container
├── Dockerfile.dashboard    # Dashboard service container
├── config.py               # Pydantic Settings (261 lines, extensive feature flags)
└── README.md               # Project overview
```

## 4. Running Locally

### Option A: Streamlit Dashboard (Recommended)

```bash
python -m streamlit run dashboard/app.py --server.port 8501
```

### Option B: Direct Pipeline Testing (No API Server)

You can also test the reasoning pipeline directly:

```python
from reasoning.neuro_symbolic_pipeline import NeuroSymbolicPipeline

pipeline = NeuroSymbolicPipeline()
result = pipeline.process_query(
    query="How do nanoparticles cross the blood-brain barrier?",
    user_role="researcher",
    session_id="demo"
)
print(result)
```

### Option C: Run the API Server

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then access the API at `http://localhost:8000` with endpoints `/api/query` (POST), `/api/ingest` (POST), `/api/stream_reasoning` (GET, SSE), `/v1/api/*` aliases, `/healthz`, `/metrics`, and `/docs`.

### Option D: Docker Compose (Full Stack)

```bash
docker compose up -d
```

This starts Redis, Qdrant, Neo4j, Jaeger, vLLM, API, Dashboard, Prometheus, and Grafana.

## 5. Key Concepts

### Execution Modes

The `UnifiedRouter` classifies queries into three execution modes:

| Mode | Complexity | Retrieval | Reasoning Model | Budget Tokens |
|------|-----------|-----------|-----------------|---------------|
| `fast` | Low / Factual | Optimized simple vector | LiteLLM / ZAYA1B simulator | 1,500 |
| `medium` | Medium | Hybrid RAG-KG | ZAYA1B | 3,500 |
| `deep` | High | Hybrid RAG-KG | ZAYA1-8B | 6,000 |

### Reasoning Paths

- **WFA Fast Path (80%)**: Decision tree + Weighted Finite Automata for high-confidence queries.
- **GraphRAG Slow Path (15%)**: Neo4j multi-hop traversal for complex cross-domain queries.
- **Abductive Deep Path (5%)**: Competing hypothesis generation for causal queries.

### Memory Architecture

- **Dual Memory**: Session context (recent interactions) + long-term domain memory.
- **DLDB**: Disk-backed lightweight database for persistent learning data.
- **DiskCache**: Persistent query result caching.

## 6. Development Workflow

1. **Run tests**: `python -m pytest tests`
   - Current state: 147 passed, no failures, no errors

2. **Validate packaging**: `python -c "import tomli; tomli.load(open('pyproject.toml','rb'))"`

3. **Validate YAML**: `python scripts/validate_yaml.py .github/workflows monitoring kubernetes helm`

4. **Run security check**: `python scripts/security_check.py --skip-pip-audit`

5. **Compile sources**: `python -m compileall app ingestion reasoning vector_store dashboard scripts migrations`

6. **Health check**: `python health_check.py`
    - Validates imports for all major components

7. **Feature flags**: Use `config.py` `Settings` class to toggle capabilities.
    - Many flags default to `True` for development, `False` for production constraints.

## 7. Common Pitfalls

- **Empty packages**: `core/`, `infra/`, `services/` contain only `__init__.py`. Do not expect implementation there; it lives in `reasoning/`, `ingestion/`, and `vector_store/`.
- **Redis/Neo4j not required for local dev**: `QDRANT_IN_MEMORY=True` and `NEO4J_ENABLED=False` let you run without external services.
- **React frontend is disabled by default**: Enable with `REACT_UI_ENABLED=True` in `.env`.
- **vLLM requires GPU**: The Docker Compose `vllm` service uses GPU reservations; it will not start on CPU-only hosts.
