# CrossMind: Neuro-Symbolic AI Scientific Discovery Engine

**CrossMind** is an optimized neuro-symbolic workflow for cross-domain scientific discovery powered by **Yuuki RxG Nano (1.5B)** as the neural reasoning brain and **Qdrant** as the secure vector retrieval layer.

---

## 🏆 Why Yuuki RxG Nano is Ideal for CrossMind

| Metric | Performance | Advantage for CrossMind |
| :--- | :--- | :--- |
| **AIME 2024** | **80.0%** | 2.77× higher than DeepSeek-R1-Distill-1.5B (28.9%) |
| **TruthfulQA MC1** | **89.6%** | High factual accuracy; eliminates scientific hallucinations |
| **MMLU-Pro** | **65.63%** | Outperforms DeepSeek V3 671B (64.4%) at 1/447th the size |
| **MMLU** | **85.4%** | Exceptional multi-task benchmark performance |
| **Training Cost** | **< $15** | Ultra cost-efficient (<90 mins on a single GPU) |
| **Memory Footprint** | **~1.5B params** | ~3-4GB VRAM peak consumption |
| **License** | **Apache 2.0** | Full commercial and research freedom |

---

## 📊 CrossMind 4-Phase Architecture Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: UNIFIED MULTIMODAL INGESTION               │
│                     FastAPI + Celery + Multimodal Embeddings           │
│                                                                         │
│  • FastAPI async endpoints for concurrent document uploads             │
│  • Celery workers for distributed embedding jobs                      │
│  • nomic-embed-text (274MB) with Matryoshka truncation (256-dim)      │
│  • Parallel writes to Qdrant                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: SECURE VECTOR RETRIEVAL                    │
│                     Qdrant + HNSW + Metadata Filtering                 │
│                                                                         │
│  • O(log N) HNSW search ~5-15ms for 1M vectors                        │
│  • Inline RBAC filtering at retrieval layer                          │
│  • 256-dim vectors with scalar quantization                           │
│  • Memory: ~1.5GB for 1M vectors                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  PHASE 3: NEURO-SYMBOLIC REASONING                     │
│            Symbolic Rules + Yuuki RxG Nano (1.5B) Agent               │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Step 3a: Symbolic Pre-Filter (<50ms)                            │  │
│  │   • Deterministic rule engine filters candidates               │  │
│  │   • Reduces agent's search space                               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    │                                   │
│                                    ▼                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Step 3b: Yuuki RxG Nano Agentic Reasoning (~2-4s)               │  │
│  │   • Native <think> blocks for explicit transparent reasoning     │  │
│  │   • 4,096 token context window                                  │  │
│  │   • Bilingual support (English / Spanish)                       │  │
│  │   • Tool-calling support                                        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    │                                   │
│                                    ▼                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Step 3c: Symbolic Post-Validation (<50ms)                       │  │
│  │   • Verifies hypotheses against scientific rules                │  │
│  │   • Reduces hallucination & checks biocompatibility              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 4: STRUCTURED STREAMING                       │
│                     FastAPI + Streamlit Dashboard                      │
│                                                                         │
│  • Real-time SSE streaming of evidence, confidence, citations         │
│  • Interactive visualization with Plotly & PyVis                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start & Execution Guide

### Option 1: Run Command-Line Demo

```bash
# Install dependencies
pip install -r requirements.txt

# Run full CrossMind workflow execution
python run_demo.py "Find cross-domain links between Alzheimer's biomarkers and nanomaterials"
```

### Option 2: Run FastAPI Server & Streamlit Dashboard

```bash
# Start FastAPI backend server (Port 8000)
uvicorn app.main:app --reload --port 8000

# Start Streamlit interactive dashboard (Port 8501)
streamlit run dashboard/app.py
```

Open browser at `http://localhost:8501`.

### Option 3: Docker Deployment

```bash
docker-compose up --build
```

---

## 🚀 Serving Yuuki RxG Nano with vLLM

To serve the actual Yuuki RxG Nano model using vLLM:

```bash
# Serve model via vLLM
vllm serve "OpceanAI/Yuuki-RxG-nano" --port 8000
```

---

## 📈 Time & Space Complexity

| Phase | Time Complexity | Space Complexity | Throughput |
| :--- | :--- | :--- | :--- |
| **Phase 1: Ingestion** | O(N) async | 1-5KB/chunk | Unlimited (Celery) |
| **Phase 2: Qdrant** | O(log N) ~5-15ms | ~1.5GB (1M vectors) | ~10K QPS |
| **Phase 3a: Pre-filter** | < 50ms | < 1MB | Instant |
| **Phase 3b: RxG Nano Reasoning** | ~2-4s | ~3-4GB VRAM | High efficiency |
| **Phase 3c: Post-Validation** | < 50ms | < 1MB | Instant |
| **Phase 4: Streaming** | Real-time SSE | < 100MB/session | Progressive |
| **Total End-to-End** | **~7-14 seconds** | **~4-5GB peak** | **Single Consumer GPU** |
