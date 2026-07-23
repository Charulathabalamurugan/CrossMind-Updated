# CrossMind: Neuro-Symbolic AI Scientific Discovery Engine

**CrossMind** is an optimized neuro-symbolic workflow for cross-domain scientific discovery powered by **Yuuki RxG Nano (1.5B)** as the neural reasoning brain and **Qdrant** as the secure vector retrieval layer.

Instead of heavy NLP embedding models, CrossMind uses **DSKE (Document-Symbolic Knowledge Embedding)** — a deterministic, domain-aware feature-hashing engine — combined with **Aho-Corasick domain keyword matching** and **BM25 + HNSW hybrid retrieval** for fast, inspectable, O(log N) search.

---

## Evidence-grounded discovery signals

Each query returns decision-support artifacts:

- **`graph_rag`**: a typed knowledge subgraph built from RBAC-filtered evidence, including supported multi-hop document → entity → document paths.
- **`cross_domain_scoring`**: an inspectable 0–100 discovery-strength score combining semantic relevance (30%), evidence coverage (25%), domain diversity (25%), and graph bridge strength (20%).
- **`confidence_calibration`**: a conservative confidence estimate derived from the model estimate (35%), discovery strength (40%), and symbolic validation (25%), with a confidence interval and recommended decision state.
- **`evidence_traceability`**: the exact most relevant sentence from every retrieved document, with matched query terms.
- **`abductive_reasoning`**: ranked causal explanations connecting biomarkers to delivery mechanisms.
- **`memory_footprint`**: MirrorMind episodic, semantic, domain, and interdisciplinary memory profile.

Graph expansion stays within the retrieved, access-controlled evidence set; it never introduces a document the requesting role could not retrieve.

---

## 🏆 Why Yuuki RxG Nano is Ideal for CrossMind

| Metric | Performance | Advantage for CrossMind |
| :--- | :--- | :--- |
| **AIME 2024** | **80.0%** | 2.77× higher than DeepSeek-R1-Distill-1.5B (28.9%) |
| **TruthfulQA MC1** | **89.6%** | High factual accuracy; reduces scientific hallucinations |
| **MMLU-Pro** | **65.63%** | Outperforms DeepSeek V3 671B (64.4%) at 1/447th the size |
| **MMLU** | **85.4%** | Exceptional multi-task benchmark performance |
| **Training Cost** | **< $15** | Ultra cost-efficient (<90 mins on a single GPU) |
| **Memory Footprint** | **~1.5B params** | ~3–4 GB VRAM peak consumption |
| **License** | **Apache 2.0** | Full commercial and research freedom |

---

## 📊 CrossMind 4-Phase Architecture Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: UNIFIED DOCUMENT INGESTION                  │
│                     FastAPI + DSKE Embedding + Qdrant Upsert             │
│                                                                         │
│  • FastAPI async endpoints for concurrent document uploads             │
│  • DSKE (Document-Symbolic Knowledge Embedding) — 64-dim vectors       │
│    (deterministic feature hashing, no GPU, ~40× faster than ST)        │
│  • Parallel writes to Qdrant + BM25 keyword index                      │
│  • Knowledge graph indexing from document tags & entities              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: SECURE HYBRID RETRIEVAL                     │
│              Qdrant HNSW + BM25 RRF + Domain/RBAC Filtering            │
│                                                                         │
│  • O(log N) HNSW dense search (~5–15 ms for 1M vectors)               │
│  • BM25 sparse keyword ranking fused via Reciprocal Rank Fusion        │
│  • Inline RBAC filtering at retrieval layer                          │
│  • 64-dim vectors with scalar quantization                             │
│  • Parallel per-domain search when multiple domains detected           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  PHASE 3: NEURO-SYMBOLIC REASONING                      │
│            Symbolic Rules + Yuuki RxG Nano (1.5B) Agent                │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Step 3a: Symbolic Pre-Filter (<50 ms)                           │  │
│  │   • Aho-Corasick automaton over domain-specific keyword ontology│  │
│  │   • O(N + M) entity & domain detection                          │  │
│  │   • English / Spanish language detection                        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    │                                   │
│                                    ▼                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Step 3b: Yuuki RxG Nano Agentic Reasoning (~2–4 s)              │  │
│  │   • Native <think> blocks for transparent reasoning │  │
│  │   • 4,096 token context window                                  │  │
│  │   • Bilingual support (English / Spanish)                       │  │
│  │   • Tool-calling support + local simulator fallback             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    │                                   │
│                                    ▼                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Step 3c: Symbolic Post-Validation (<50 ms)                      │  │
│  │   • Biocompatibility, temporal alignment, citation grounding    │  │
│  │   • Reduces hallucination & checks scientific constraints       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    │                                   │
│                                    ▼                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Step 4: GraphRAG + Abductive + Memory Enrichment                │  │
│  │   • Multi-hop cross-domain path scoring                         │  │
│  │   • Causal explanation candidates                               │  │
│  │   • MirrorMind research memory profile                          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 4: STRUCTURED STREAMING                        │
│                     FastAPI + Streamlit Dashboard                       │
│                                                                         │
│  • Real-time SSE streaming of evidence, confidence, citations          │
│  • Interactive GraphRAG network with Plotly                           │
│  • Confidence policy sliders (Proceed / Investigate thresholds)        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 DSKE Embedding Engine

CrossMind replaces SentenceTransformer / nomic-embed-text with **DSKE (Document-Symbolic Knowledge Embedding)**:

| Property | DSKE | Traditional NLP Embedding |
| :--- | :--- | :--- |
| **Speed** | ~40× faster | GPU-bound inference |
| **Memory** | ~50× less | 274 MB+ model weights |
| **Determinism** | Same text → same vector | Model/version dependent |
| **Domain control** | Combined with Aho-Corasick ontology | Black-box semantic similarity |
| **Default dimension** | 64 (configurable via `EMBEDDING_DIM`) | 256–768 typical |

DSKE hashes each word into a fixed-size vector with position weighting and L2 normalization. At query time it is fused with BM25 keyword scores and Qdrant HNSW dense search for hybrid O(log N) retrieval.

---

## 📁 Project Structure

```
CrossMind-Updated/
├── app/
│   ├── main.py              # FastAPI routes, security middleware, startup seeding
│   └── observability.py     # JSON logging + Prometheus metrics
├── config.py                # Environment-driven settings
├── dashboard/
│   └── app.py               # Streamlit UI (reasoning, benchmarks, ingestion)
├── ingestion/
│   ├── embedding.py         # DSKE embedder
│   ├── pipeline.py          # Document ingestion pipeline
│   └── seed_data.py         # Default scientific knowledge base (6 papers)
├── reasoning/
│   ├── neuro_symbolic_pipeline.py  # Main orchestrator
│   ├── symbolic_filter.py            # Aho-Corasick pre/post validation
│   ├── rxg_nano_agent.py             # Yuuki RxG Nano agent + simulator
│   ├── knowledge_graph.py            # GraphRAG + discovery scoring
│   ├── abductive_engine.py           # Causal explanation engine
│   ├── memory_service.py             # MirrorMind memory framework
│   └── traceability.py               # Evidence passage selection
├── vector_store/
│   └── qdrant_engine.py     # Qdrant HNSW + BM25 hybrid search + RBAC
├── comprehensive_test.py    # 48-test API & pipeline suite
├── thorough_testing.py      # 52-test CORS, concurrency, dashboard, Docker suite
├── run_api.bat              # Windows: start FastAPI on port 8000
└── run_streamlit.bat        # Windows: start Streamlit on port 8501
```

---

## ⚡ Quick Start & Execution Guide

### Prerequisites

```bash
pip install -r requirements.txt
```

Copy environment defaults if needed:

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `QDRANT_IN_MEMORY` | `True` | Use in-process Qdrant (no external server needed) |
| `EMBEDDING_DIM` | `64` | DSKE vector dimension |
| `USE_LOCAL_SIMULATOR_FALLBACK` | `True` | Use built-in RxG Nano simulator when vLLM unavailable |
| `API_BASE` | `http://localhost:8000` | Backend URL for Streamlit dashboard |

---

### Option 1: Run Command-Line Demo

```bash
python run_demo.py "Find cross-domain links between Alzheimer's biomarkers and nanomaterials"
```

---

### Option 2: Run FastAPI Server & Streamlit Dashboard (Recommended)

**Terminal 1 — API (port 8000):**

```bash
# Linux / macOS
QDRANT_IN_MEMORY=True uvicorn app.main:app --host 127.0.0.1 --port 8000

# Windows PowerShell
$env:QDRANT_IN_MEMORY="True"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Or on Windows, double-click **`run_api.bat`**.

**Terminal 2 — Streamlit dashboard (port 8501):**

```bash
# Linux / macOS
API_BASE=http://127.0.0.1:8000 streamlit run dashboard/app.py --server.port 8501

# Windows PowerShell
$env:API_BASE="http://127.0.0.1:8000"
python -m streamlit run dashboard/app.py --server.port 8501
```

Or on Windows, double-click **`run_streamlit.bat`**.

Open **http://localhost:8501**, select a sample query, and click **Run CrossMind Workflow**.

---

### Option 3: Docker Deployment

```bash
docker compose up --build
```

| Service | URL |
| :--- | :--- |
| Dashboard | http://localhost:8501 |
| API | http://localhost:8000 |
| Qdrant | http://localhost:6333 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (default: `admin` / `admin`) |

The API emits JSON request logs, exposes `GET /healthz` for health checks, and Prometheus metrics at `GET /metrics`.

---

## 🧪 Running Tests

Start the API first, then run both suites:

```bash
python comprehensive_test.py   # 48 tests — endpoints, queries, RBAC, SSE, ingestion
python thorough_testing.py     # 52 tests — CORS, latency, concurrency, dashboard deps, Docker
```

Expected result: **100% pass rate** on both suites when the API is running at `http://127.0.0.1:8000`.

---

## 🔌 API Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Service info and endpoint list |
| `GET` | `/healthz` | Health check |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/api/metrics` | Benchmark numbers for dashboard charts |
| `POST` | `/api/query` | Full neuro-symbolic query pipeline |
| `POST` | `/api/ingest` | Ingest documents into Qdrant |
| `GET` | `/api/stream_reasoning` | SSE streaming version of the pipeline |

**Example query:**

```bash
curl -X POST http://127.0.0.1:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Find cross-domain links between Alzheimer biomarkers and nanomaterials", "user_role": "researcher"}'
```

---

## 🚀 Serving Yuuki RxG Nano with vLLM (Optional)

By default CrossMind uses a built-in simulator. To connect a live model:

```bash
vllm serve "OpceanAI/Yuuki-RxG-nano" --port 8001
```

Then set in `.env`:

```
RXG_NANO_API_BASE=http://localhost:8001/v1
USE_LOCAL_SIMULATOR_FALLBACK=False
```

---

## 📈 Time & Space Complexity

| Phase | Time Complexity | Space Complexity | Notes |
| :--- | :--- | :--- | :--- |
| **Phase 1: Ingestion** | O(words) per doc | ~1–5 KB/chunk | DSKE hashing, no model load |
| **Phase 2: Qdrant HNSW** | O(log N) ~5–15 ms | ~1.5 GB (1M vectors) | + BM25 RRF fusion |
| **Phase 3a: Pre-filter** | O(N + M) < 50 ms | < 1 MB | Aho-Corasick automaton |
| **Phase 3b: RxG Nano** | ~2–4 s | ~3–4 GB VRAM (live) | Simulator: no GPU |
| **Phase 3c: Post-Validation** | < 50 ms | < 1 MB | Rule engine |
| **Phase 4: Streaming** | Real-time SSE | < 100 MB/session | Progressive delivery |
| **Total End-to-End** | **~5–14 s** | **~1.5 GB RAM (simulator mode)** | Single consumer machine |

---

## 🔒 Security Features

- Optional Bearer token auth (`API_KEY` env var; disabled in dev mode when unset)
- CORS restricted to Streamlit origins
- Rate limiting per IP
- Request size limits and XSS input sanitization
- RBAC at vector retrieval layer (`public`, `researcher`, `admin`)

---

## License

Apache 2.0 — Yuuki RxG Nano model and CrossMind application code.
