# CrossMind Sequence Flows

## Phase 1: Ingestion Pipeline

```mermaid
sequenceDiagram
    participant Client
    participant IngestionPipeline
    participant TextExtractor
    participant Chunker
    participant Embedder as BGE-M3 Embedder
    participant VectorAdapter
    participant Qdrant
    participant BM25
    participant Cache as IngestionCache

    Client->>IngestionPipeline: ingest_documents(docs)
    IngestionPipeline->>Cache: check content_hash
    alt cache hit
        Cache-->>IngestionPipeline: skip duplicate
    else cache miss
        IngestionPipeline->>TextExtractor: extract(file_path/content)
        TextExtractor-->>IngestionPipeline: raw_text
        IngestionPipeline->>Chunker: chunk_text(text, metadata)
        Chunker-->>IngestionPipeline: chunks[]
        IngestionPipeline->>Embedder: embed_texts(chunks)
        Embedder-->>IngestionPipeline: dense_vectors[]
        IngestionPipeline->>VectorAdapter: normalize(vectors)
        VectorAdapter-->>IngestionPipeline: normalized_vectors[]
        loop for each chunk
            IngestionPipeline->>Qdrant: upsert(collection, points)
            IngestionPipeline->>BM25: index_document(id, text)
        end
        IngestionPipeline->>Cache: set(content_hash, True)
    end
    IngestionPipeline-->>Client: inserted_ids[]
```

## Phase 2: Hybrid Retrieval

```mermaid
sequenceDiagram
    participant Pipeline as NeuroSymbolicPipeline
    participant Classifier as QueryClassifier
    participant Router as UnifiedRouter
    participant Embedder as BGE-M3 Embedder
    participant Adapter as VectorAdapter
    participant Qdrant
    participant BM25Engine
    participant RRF as ReciprocalRankFusion
    participant Cache as QueryCache

    Pipeline->>Classifier: classify(query)
    Classifier-->>Pipeline: complexity, domain, query_type
    Pipeline->>Router: route(query, metadata)
    Router-->>Pipeline: execution_mode, model, budget_tokens
    alt fast path
        Pipeline->>Embedder: embed_text(query, dim=256)
        Embedder-->>Pipeline: query_vector
        Pipeline->>Adapter: normalize(query_vector)
        Adapter-->>Pipeline: normalized_vector
        Pipeline->>Qdrant: search_with_rbac(vector, role, domains)
        Qdrant-->>Pipeline: dense_results[]
    else medium/deep path
        Pipeline->>Embedder: embed_text(query)
        Embedder-->>Pipeline: query_vector
        Pipeline->>Qdrant: search_with_rbac(vector, role, domains)
        Qdrant-->>Pipeline: dense_results[]
        Pipeline->>BM25Engine: search(query, domains, role)
        BM25Engine-->>Pipeline: sparse_results[]
        Pipeline->>RRF: fuse(dense_results, sparse_results)
        RRF-->>Pipeline: fused_results[]
    end
    Pipeline->>Cache: check(query, role, session)
    alt cache hit
        Cache-->>Pipeline: cached_result
    end
    Pipeline-->>Pipeline: retrieved_evidence[]
```

## Phase 3: Neuro-Symbolic Reasoning

```mermaid
sequenceDiagram
    participant Pipeline as NeuroSymbolicPipeline
    participant PreFilter as SymbolicPreFilter
    participant WFA as WFAFastPath
    participant DT as DecisionTree
    participant GraphRAG as HybridRAGKG
    participant MultiAgent as MultiAgentOrchestrator
    participant Agent as ZAYA1-8B Agent
    participant Z3 as Z3Validator
    participant PostVal as SymbolicPostValidator

    Pipeline->>PreFilter: process(query)
    PreFilter-->>Pipeline: domains, entities, lang
    Pipeline->>Pipeline: route_reasoning_model(complexity)
    alt fast path (WFA + DecisionTree)
        Pipeline->>WFA: evaluate(query, evidence)
        WFA-->>Pipeline: fast_result
        Pipeline->>DT: classify(query)
        DT-->>Pipeline: decision, confidence
    else deep path (GraphRAG + Multi-Agent)
        Pipeline->>GraphRAG: retrieve(query, domains)
        GraphRAG-->>Pipeline: graph_context, paths
        Pipeline->>MultiAgent: parallel_process(query, evidence, metadata)
        MultiAgent-->>Pipeline: agent_reports[]
        Pipeline->>Agent: reason_and_synthesize(query, evidence, context)
        Agent-->>Pipeline: hypothesis, confidence, citations
    end
    Pipeline->>PostVal: validate(hypothesis, evidence)
    PostVal-->>Pipeline: validation_score, passed
    Pipeline->>Z3: validate_hypothesis(agent_result, evidence)
    Z3-->>Pipeline: z3_score, z3_passed
    Pipeline-->>Pipeline: enriched_result
```

## Phase 4: Enrichment & Output

```mermaid
sequenceDiagram
    participant Pipeline as NeuroSymbolicPipeline
    participant KG as KnowledgeGraph
    participant Scorer as DiscoveryScorer
    participant Calibrator as ConfidenceCalibrator
    participant Attributor as EvidenceAttributor
    participant Blueprint as ExperimentalBlueprint
    participant Collab as CollaborationRecommender
    participant Memory as DualMemory
    participant SSE as StreamingSSE

    Pipeline->>KG: graph_rag_context(evidence, entities)
    KG-->>Pipeline: graph_context
    Pipeline->>Scorer: score(evidence, graph_context)
    Scorer-->>Pipeline: discovery_score, rating
    Pipeline->>Calibrator: calibrate(confidence, discovery, validation)
    Calibrator-->>Pipeline: calibrated_confidence
    Pipeline->>Attributor: attribute(hypothesis, evidence)
    Attributor-->>Pipeline: attribution_coverage, supported_claims
    Pipeline->>Blueprint: generate(agent_result, abductive, evidence)
    Blueprint-->>Pipeline: blueprint (title, objective, timeline)
    Pipeline->>Collab: recommend(paths, domains)
    Collab-->>Pipeline: recommendations[]
    Pipeline->>Memory: update_session(session_id, interaction)
    Pipeline->>SSE: stream_events(stage, data)
    SSE-->>Client: SSE chunks (thinking, hypothesis, validation, blueprint)
    Pipeline-->>Client: JSON response
```

## End-to-End Request Flow (Including Frontend)

```mermaid
flowchart TD
    A["User Query"] --> B{UI Path}
    B -->|Streamlit| C["dashboard/app.py<br/>call_api('/api/query')"]
    B -->|React (optional)| D["frontend/src/phases/<br/>fetch('/api/query')"]
    B -->|Direct| E["HTTP POST /api/query"]

    C --> F["FastAPI App<br/>(app/main.py:343 lines)"]
    D --> F
    E --> F

    F --> G["NeuroSymbolicPipeline.process_query()"]
    G --> H["SymbolicPreFilter"]
    H --> I["UnifiedRouter.route()"]
    I --> J{"Execution Mode"}
    J -->|fast| K["WFA + DecisionTree"]
    J -->|medium/deep| L["GraphRAG + MultiAgent + ZAYA1-8B"]
    K --> M["Z3 + PostValidation"]
    L --> M
    M --> N["Enrichment + Memory + Streaming"]
    N --> O["JSON / SSE Response"]

    style F fill:#ccffcc,stroke:#333,stroke-width:2px
    style O fill:#ccffcc,stroke:#333,stroke-width:2px
```

> **Note**: The FastAPI application layer in `app/main.py` is implemented and exports a real `app` object. The HTTP endpoints `/api/query`, `/api/ingest`, `/api/stream_reasoning`, and `/v1/api/*` are bound and active. The sequence diagrams below describe the full flow including the API layer.
