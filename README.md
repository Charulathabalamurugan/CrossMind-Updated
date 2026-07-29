# CrossMind: Neuro-Symbolic Scientific Discovery Engine (Prototype)

**CrossMind** is a prototype neuro-symbolic AI system for cross-domain scientific discovery powered by **ZAYA1-8B** (8.4B MoE, 760M active parameters) as the reasoning engine and **BGE-M3** + **Qdrant** for retrieval.

## Tech Stack

| Phase | Technology | Memory | Latency |
|:---|:---|:---|:---|
| **1: Retrieval** | BGE-M3 + Qdrant (PQ) + Redis + MinerU + Tika | 1.8GB | 10-15ms |
| **2: Validation** | GLiNER + Unified Ontology + Soufflé Datalog + OPA | 380MB | 45ms |
| **3: Reasoning** | WFA + Decision Tree + GraphRAG + **ZAYA1-8B** | ~6GB | 1-2s |
| **4: Learning** | FastAPI SSE + Streamlit + Prometheus + Redis + DLDB | 350MB | <1ms |
| **TOTAL** | | **~8.5GB** | **~1-2s** |

## Key Components

- **ZAYA1-8B** — 8.4B MoE model, 760M active params, Q4_K_M quantized (~5.5GB), 131K context, Apache 2.0 license, 89.1% AIME 2026
- **BGE-M3** — INT8 quantized embedding, dense+sparse, O(log N) retrieval via Qdrant HNSW
- **ZAYA1-8B Agent** — Native `[THINK][/THINK]` reasoning blocks, Markovian RSA, compressed attention, vLLM serving
- **Streamlit Dashboard** — Single-page 6-phase flow, runs all phases sequentially on one click
- **FastAPI Backend** — 9 endpoints, SSE streaming, RBAC, Prometheus metrics

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

Open http://localhost:8501, enter a query, click **Run Full Pipeline**.

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