# CrossMind: Neuro-Symbolic Scientific Discovery Engine (Prototype)

**CrossMind** is a prototype neuro-symbolic AI system for cross-domain scientific discovery powered by **ZAYA1-8B** (8.4B MoE, 760M active) as the deep reasoning engine with **BGE-M3** + **Qdrant** for retrieval, **ZAYA-1B** for moderate reasoning, and **LiteLLM** for fast factual queries.

## Phase 1: Ingestion

**Technologies:** FastAPI, MinerU, Apache Tika, BGE-M3, Qdrant, Redis

**Purpose:** Collects and processes documents, extracts text and structured content, generates dual-vector Matryoshka embeddings (compact 256-dim search + full 1024-dim rerank), stores them in the vector database, and uses caching for efficient data ingestion.

- FastAPI async endpoints for document upload (PDF, DOCX, Excel, Email)
- MinerU for scientific PDF extraction (tables, formulas, images)
- Apache Tika as fallback for office docs and emails
- BGE-M3 INT8 or FP32 quantized with Matryoshka support (256-dim search vector + 1024-dim rerank vector)
- Qdrant with PQ compression for vector storage
- Redis for hot-query caching (sub-2ms)
- Domain classification on ingest

## Phase 2: Retrieval

**Technologies:** BGE-M3, Qdrant, BM25, RRF, Cross-Encoder Reranking, RBAC, Redis, LightGBM/TinyBERT Classifier, Conditional Retrieval

**Purpose:** Performs hybrid semantic and keyword-based retrieval, reranks results using a lightweight cross-encoder (replacing ColBERT), applies role-based access control, uses semantic caching with cleanup, and employs a machine learning query classifier to conditionally optimize retrieval pathways.

- LightGBM / TinyBERT Query Classifier (via TfidfVectorizer + GradientBoosting/MLP) categorizing query domains, types, and complexities
- Conditional Retrieval Optimization bypassing heavy multi-agent / GraphRAG steps for low-complexity / factual queries to minimize latency
- BGE-M3 dense vector search via Qdrant HNSW index (O(log N)) using 256-dim Matryoshka vectors
- BM25 sparse keyword ranking
- Reciprocal Rank Fusion (RRF) for hybrid result merging
- Lightweight cross-encoder reranking (replaces ColBERT) using stored full-dimension vectors + lexical scoring
- Inline RBAC filtering at retrieval layer
- Redis tiered caching (hot + warm) with semantic query cache cleanup on expiry and deletion

## Phase 3: Reasoning

**Technologies:** ZAYA1-8B, ZAYA-1B, LiteLLM, vLLM, Scallop, Semara, DeforestVIS, GraphRAG, WFA, Decision Tree, Redis

**Purpose:** Performs neuro-symbolic reasoning with lightweight model routing (LiteLLM for low complexity, ZAYA-1B for moderate, ZAYA1-8B for deep reasoning), evidence compression to keep query-relevant sentences before downstream reasoning, and combines fast rule-based reasoning, ontology-based semantic reasoning, graph-based multi-hop discovery, and LLM-based reasoning with explainability and persistent context.

- Query classification routes to the appropriate model: LiteLLM (low), ZAYA-1B (medium), ZAYA1-8B (high)
- Evidence compression extracts query-relevant sentences before reasoning to reduce context overhead
- WFA + Decision Tree for fast path (80% of queries, O(1), <10ms)
- GraphRAG for slow path (15%, multi-hop graph traversal)
- ZAYA1-8B (Q4_K_M, 5.5GB, 760M active params) for deep reasoning
- ZAYA-1B for moderate-complexity reasoning
- LiteLLM (lite-llm/mini) for low-complexity / factual queries
- vLLM for continuous batching inference with `--max-num-seqs 2` to prevent OOM on RTX 4090
- Native `[THINK][/THINK]` reasoning blocks for transparency
- Markovian RSA for unbounded reasoning with fixed memory
- Scallop for logical reasoning integration
- Semara for semantic grounding (Tech Mahindra SEMARA reference; open-source SeMRA also supported via `SEMARA_IMPL` env var)
- DeforestVIS for reasoning visualization
- Redis for caching expensive reasoning results

## Phase 4: Application

**Technologies:** FastAPI, React/Streamlit, SSE, OpenTelemetry, Prometheus, Redis + DiskCache, DLDB, RBAC, Evaluation Framework

**Purpose:** Provides the user interface and real-time streaming, monitors system performance, manages hot and warm caching, stores feedback and long-term knowledge, ensures secure role-based access, and evaluates search retrieval quality.

- Retrieval Performance Evaluation Framework calculating Precision@K, Recall@K, MRR, and NDCG@K against standard ground-truth datasets
- FastAPI SSE for real-time token/citation streaming
- Streamlit prototype UI + React for enterprise production
- OpenTelemetry + Jaeger for distributed tracing
- Prometheus + Grafana for monitoring and dashboards
- Redis (hot) + DiskCache (warm) for tiered caching
- DLDB for feedback, rules, and validation results storage
- RBAC at every layer (ingestion, retrieval, reasoning, application)

## Prototype Summary

| Phase | Tech Stack | Memory | Latency |
|:---|:---|:---|:---|
| **1: Ingestion** | FastAPI + MinerU + Tika + BGE-M3 (Matryoshka) + Qdrant + Redis | 1.8GB | 10-15ms |
| **2: Retrieval** | BGE-M3 (256-dim) + Qdrant + BM25 + RRF + Cross-Encoder Reranking + RBAC + Redis | 1.8GB | 10-15ms |
| **3: Reasoning** | LiteLLM (low) / ZAYA-1B (medium) / ZAYA1-8B (high) + vLLM + Scallop + Semara + DeforestVIS + GraphRAG + WFA + Decision Tree + Redis | ~6GB | 1-2s |
| **4: Application** | FastAPI + React/Streamlit + SSE + OpenTelemetry + Prometheus + Redis + DiskCache + DLDB + RBAC | 350MB | <1ms |
| **TOTAL** | **All Phases** | **~8.5GB** | **~1-2s** |

## ZAYA1-8B Specs

| Property | Value |
|:---|:---|
| Total Parameters | 8.4B |
| Active Parameters | 760M (MoE) |
| Memory (Q4_K_M) | ~5.5 GB |
| Context Length | 131,072 tokens |
| License | Apache 2.0 |
| AIME 2026 | 89.1% |

## Quick Start

```bash
pip install -r requirements.txt
```

**Terminal 1 — API:**
```bash
$env:QDRANT_IN_MEMORY="True"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — Dashboard:**
```bash
$env:API_BASE="http://127.0.0.1:8000"
python -m streamlit run dashboard/app.py --server.port 8501
```

Open http://localhost:8501, enter a query, click **Run Full Pipeline**. All 4 phases execute in a single flow on one page.

## Cross-Domain Support

The architecture is domain-agnostic — works for energy, finance, materials, climate, psychology, and any scientific domain. The ZAYA1-8B agent handles any query generically with evidence-limited fallbacks when ontology entities are not found.

## Running Tests

```bash
python comprehensive_test.py
```

## Serving ZAYA1-8B Live (Optional)

```bash
vllm serve "ZAYA1-8B" --port 8001
```
Then set `ZAYA1_8B_API_BASE=http://localhost:8001/v1` and `USE_LOCAL_SIMULATOR_FALLBACK=False`.

## License

Apache 2.0