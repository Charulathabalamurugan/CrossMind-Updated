import os
import secrets

try:
    from pydantic_settings import BaseSettings
except ImportError:
    try:
        from pydantic import BaseSettings
    except ImportError:
        class BaseSettings:
            pass

class Settings(BaseSettings):
    PROJECT_NAME: str = "CrossMind"
    VERSION: str = "1.0.0"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # ========== Security Settings ==========
    API_KEY: str = os.getenv("API_KEY", "")
    _generated_key: str = ""

    @property
    def effective_api_key(self) -> str:
        if self.API_KEY:
            return self.API_KEY
        if not self._generated_key:
            self._generated_key = secrets.token_urlsafe(32)
        return self._generated_key

    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501,http://localhost:8000,http://127.0.0.1:8000")

    @property
    def allowed_origins_list(self) -> list:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    MAX_REQUEST_SIZE_MB: int = int(os.getenv("MAX_REQUEST_SIZE_MB", "10"))
    MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "5000"))
    MAX_DOC_CONTENT_LENGTH: int = int(os.getenv("MAX_DOC_CONTENT_LENGTH", "50000"))
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "1000"))

    # ========== ZAYA1-8B Agent settings ==========
    ZAYA1_8B_MODEL_NAME: str = os.getenv("ZAYA1_8B_MODEL_NAME", "ZAYA1-8B")
    ZAYA1_8B_API_BASE: str = os.getenv("ZAYA1_8B_API_BASE", "http://localhost:8000/v1")
    ZAYA1_8B_TEMPERATURE: float = float(os.getenv("ZAYA1_8B_TEMPERATURE", "0.2"))
    ZAYA1_8B_MAX_TOKENS: int = int(os.getenv("ZAYA1_8B_MAX_TOKENS", "131072"))
    ZAYA1_8B_TOTAL_PARAMS: int = int(os.getenv("ZAYA1_8B_TOTAL_PARAMS", "8400000000"))
    ZAYA1_8B_ACTIVE_PARAMS: int = int(os.getenv("ZAYA1_8B_ACTIVE_PARAMS", "760000000"))
    ZAYA1_8B_QUANTIZATION: str = os.getenv("ZAYA1_8B_QUANTIZATION", "Q4_K_M")
    ZAYA1_8B_MEMORY_FOOTPRINT_GB: float = float(os.getenv("ZAYA1_8B_MEMORY_FOOTPRINT_GB", "5.5"))
    ZAYA1_8B_CONTEXT_LENGTH: int = int(os.getenv("ZAYA1_8B_CONTEXT_LENGTH", "131072"))
    ZAYA1_8B_LICENSE: str = os.getenv("ZAYA1_8B_LICENSE", "Apache 2.0")
    ZAYA1_8B_AIME_2026_SCORE: float = float(os.getenv("ZAYA1_8B_AIME_2026_SCORE", "89.1"))
    ZAYA1_8B_MOE_ARCHITECTURE: bool = os.getenv("ZAYA1_8B_MOE_ARCHITECTURE", "True").lower() == "true"
    ZAYA1_8B_MARKOVIAN_RSA: bool = os.getenv("ZAYA1_8B_MARKOVIAN_RSA", "True").lower() == "true"
    ZAYA1_8B_COMPRESSED_ATTENTION: bool = os.getenv("ZAYA1_8B_COMPRESSED_ATTENTION", "True").lower() == "true"
    ZAYA1_8B_NATIVE_THINK_BLOCKS: bool = os.getenv("ZAYA1_8B_NATIVE_THINK_BLOCKS", "True").lower() == "true"
    ZAYA1_8B_VLLM_ENABLED: bool = os.getenv("ZAYA1_8B_VLLM_ENABLED", "False").lower() == "true"
    ZAYA1_8B_VLLM_BATCH_SIZE: int = int(os.getenv("ZAYA1_8B_VLLM_BATCH_SIZE", "8"))
    USE_LOCAL_SIMULATOR_FALLBACK: bool = os.getenv("USE_LOCAL_SIMULATOR_FALLBACK", "True").lower() == "true"

    # ========== Embedding settings ==========
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "nomic-ai/nomic-embed-text-v1.5")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "1024"))

    # ========== Phase 1: Ingestion Settings ==========
    # FastAPI (backend framework)
    # MinerU extraction
    MINERU_EXTRACTION_ENABLED: bool = os.getenv("MINERU_EXTRACTION_ENABLED", "True").lower() == "true"
    MINERU_API_BASE: str = os.getenv("MINERU_API_BASE", "http://localhost:8002")
    # Apache Tika fallback
    TIKA_FALLBACK_ENABLED: bool = os.getenv("TIKA_FALLBACK_ENABLED", "True").lower() == "true"
    TIKA_SERVER_URL: str = os.getenv("TIKA_SERVER_URL", "http://localhost:9998")
    # BGE-M3 embedding
    BGE_M3_ENABLED: bool = os.getenv("BGE_M3_ENABLED", "True").lower() == "true"
    BGE_M3_MODEL_NAME: str = os.getenv("BGE_M3_MODEL_NAME", "BAAI/bge-m3")
    BGE_M3_DIM: int = int(os.getenv("BGE_M3_DIM", "1024"))
    BGE_M3_RETRIEVAL_DIM: int = int(os.getenv("BGE_M3_RETRIEVAL_DIM", "256"))
    BGE_M3_MATRYOSHKA_ENABLED: bool = os.getenv("BGE_M3_MATRYOSHKA_ENABLED", "True").lower() == "true"
    BGE_M3_PRECISION: str = os.getenv("BGE_M3_PRECISION", "fp32")
    BGE_M3_MAX_LENGTH: int = int(os.getenv("BGE_M3_MAX_LENGTH", "8192"))
    # Qdrant vector store
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_IN_MEMORY: bool = os.getenv("QDRANT_IN_MEMORY", "True").lower() == "true"
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "crossmind_knowledge")
    PRODUCT_QUANTIZATION_ENABLED: bool = os.getenv("PRODUCT_QUANTIZATION_ENABLED", "True").lower() == "true"
    DOMAIN_QDRANT_COLLECTIONS: bool = os.getenv("DOMAIN_QDRANT_COLLECTIONS", "False").lower() == "true"
    # Redis caching
    REDIS_QUERY_CACHE_TTL: int = int(os.getenv("REDIS_QUERY_CACHE_TTL", "3600"))
    REDIS_QUERY_CACHE_MAX: int = int(os.getenv("REDIS_QUERY_CACHE_MAX", "5000"))

    # ========== Phase 2: Retrieval Settings ==========
    BGE_M3_RETRIEVAL_ENABLED: bool = os.getenv("BGE_M3_RETRIEVAL_ENABLED", "True").lower() == "true"
    # BM25 sparse retrieval
    BM25_ENABLED: bool = os.getenv("BM25_ENABLED", "True").lower() == "true"
    BM25_K1: float = float(os.getenv("BM25_K1", "1.2"))
    BM25_B: float = float(os.getenv("BM25_B", "0.75"))
    # RRF fusion
    RRF_ENABLED: bool = os.getenv("RRF_ENABLED", "True").lower() == "true"
    RRF_K: int = int(os.getenv("RRF_K", "60"))
    # ColBERT reranking (server-side via Qdrant API)
    COLBERT_RERANKING_ENABLED: bool = os.getenv("COLBERT_RERANKING_ENABLED", "True").lower() == "true"
    COLBERT_MODEL_NAME: str = os.getenv("COLBERT_MODEL_NAME", "colbert-ir/colbertv2.0")
    COLBERT_TOP_K: int = int(os.getenv("COLBERT_TOP_K", "20"))
    COLBERT_THRESHOLD: float = float(os.getenv("COLBERT_THRESHOLD", "0.7"))
    # Qdrant MultiVectorConfig for ColBERT (MAX_SIM similarity, m=0 quantization)
    COLBERT_MULTIVECTOR_ENABLED: bool = os.getenv("COLBERT_MULTIVECTOR_ENABLED", "True").lower() == "true"
    COLBERT_MULTIVECTOR_MAX_SIM: str = os.getenv("COLBERT_MULTIVECTOR_MAX_SIM", "MAX_SIM")
    COLBERT_MULTIVECTOR_M: int = int(os.getenv("COLBERT_MULTIVECTOR_M", "0"))
    CROSS_ENCODER_RERANKING_ENABLED: bool = os.getenv("CROSS_ENCODER_RERANKING_ENABLED", "True").lower() == "true"
    CROSS_ENCODER_TOP_K: int = int(os.getenv("CROSS_ENCODER_TOP_K", "20"))
    CROSS_ENCODER_MAX_CANDIDATES: int = int(os.getenv("CROSS_ENCODER_MAX_CANDIDATES", "40"))
    SEMANTIC_QUERY_CACHE_THRESHOLD: float = float(os.getenv("SEMANTIC_QUERY_CACHE_THRESHOLD", "0.92"))
    SEMANTIC_QUERY_CACHE_DIM: int = int(os.getenv("SEMANTIC_QUERY_CACHE_DIM", "256"))
    # RBAC at retrieval layer
    RBAC_ENABLED: bool = os.getenv("RBAC_ENABLED", "True").lower() == "true"
    RBAC_ROLES: str = os.getenv("RBAC_ROLES", "admin,analyst,viewer")
    # Redis tiered caching for retrieval
    REDIS_RETRIEVAL_CACHE_TTL: int = int(os.getenv("REDIS_RETRIEVAL_CACHE_TTL", "1800"))
    REDIS_RETRIEVAL_CACHE_MAX: int = int(os.getenv("REDIS_RETRIEVAL_CACHE_MAX", "10000"))

    # ========== Phase 3: Reasoning Settings ==========
    NEO4J_ENABLED: bool = os.getenv("NEO4J_ENABLED", "False").lower() == "true"
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")
    # WFA + Decision Tree fast path
    WFA_FAST_PATH_ENABLED: bool = os.getenv("WFA_FAST_PATH_ENABLED", "True").lower() == "true"
    FAST_PATH_CONFIDENCE_THRESHOLD: float = float(os.getenv("FAST_PATH_CONFIDENCE_THRESHOLD", "0.85"))
    DECISION_TREE_ENABLED: bool = os.getenv("DECISION_TREE_ENABLED", "True").lower() == "true"
    # LiteLLM routing for moderate complexity reasoning
    LITELLM_ENABLED: bool = os.getenv("LITELLM_ENABLED", "True").lower() == "true"
    LITELLM_MODEL_NAME: str = os.getenv("LITELLM_MODEL_NAME", "lite-llm/mini")
    # GraphRAG slow path
    GRAPH_RAG_ENABLED: bool = os.getenv("GRAPH_RAG_ENABLED", "True").lower() == "true"
    GRAPH_RAG_DEPTH: int = int(os.getenv("GRAPH_RAG_DEPTH", "3"))
    # vLLM serving (prevent OOM on RTX 4090)
    VLLM_ENABLED: bool = os.getenv("VLLM_ENABLED", "False").lower() == "true"
    VLLM_BATCH_SIZE: int = int(os.getenv("VLLM_BATCH_SIZE", "8"))
    VLLM_MAX_NUM_SEQS: int = int(os.getenv("VLLM_MAX_NUM_SEQS", "2"))
    VLLM_GPU_MEMORY_UTILIZATION: float = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
    VLLM_MAX_MODEL_LEN: int = int(os.getenv("VLLM_MAX_MODEL_LEN", "131072"))
    # ZAYA1-8B deep reasoning
    ZAYA1_8B_REASONING_ENABLED: bool = os.getenv("ZAYA1_8B_REASONING_ENABLED", "True").lower() == "true"
    # Scallop logical reasoning
    SCALLOP_ENABLED: bool = os.getenv("SCALLOP_ENABLED", "False").lower() == "true"
    SCALLOP_PROGRAM: str = os.getenv("SCALLOP_PROGRAM", "")
    # Semara semantic grounding (Tech Mahindra SEMARA reference implementation)
    SEMARA_ENABLED: bool = os.getenv("SEMARA_ENABLED", "True").lower() == "true"
    SEMARA_IMPL: str = os.getenv("SEMARA_IMPL", "tech-mahindra")
    SEMARA_ONTOLOGY_URL: str = os.getenv("SEMARA_ONTOLOGY_URL", "")
    SEMARA_OPEN_SOURCE_FALLBACK: bool = os.getenv("SEMARA_OPEN_SOURCE_FALLBACK", "True").lower() == "true"
    # DeforestVIS reasoning visualization
    DEFORESTVIS_ENABLED: bool = os.getenv("DEFORESTVIS_ENABLED", "False").lower() == "true"
    DEFORESTVIS_PORT: int = int(os.getenv("DEFORESTVIS_PORT", "8003"))
    # Abductive reasoning
    ABDUCTIVE_TOP_K: int = int(os.getenv("ABDUCTIVE_TOP_K", "5"))

    # ========== Phase 4: Application Settings ==========
    # React/Streamlit UI
    STREAMLIT_DASHBOARD_ENABLED: bool = os.getenv("STREAMLIT_DASHBOARD_ENABLED", "True").lower() == "true"
    REACT_UI_ENABLED: bool = os.getenv("REACT_UI_ENABLED", "False").lower() == "true"
    # SSE streaming
    STREAMING_SSE_ENABLED: bool = os.getenv("STREAMING_SSE_ENABLED", "True").lower() == "true"
    # OpenTelemetry
    OPENTELEMETRY_ENABLED: bool = os.getenv("OPENTELEMETRY_ENABLED", "True").lower() == "true"
    OPENTELEMETRY_EXPORTER: str = os.getenv("OPENTELEMETRY_EXPORTER", "http://localhost:4317")
    # Prometheus monitoring
    PROMETHEUS_ENABLED: bool = os.getenv("PROMETHEUS_ENABLED", "True").lower() == "true"
    PROMETHEUS_PORT: int = int(os.getenv("PROMETHEUS_PORT", "9090"))
    # Redis (hot) + DiskCache (warm)
    HOT_CACHE_ENABLED: bool = os.getenv("HOT_CACHE_ENABLED", "True").lower() == "true"
    DISK_CACHE_ENABLED: bool = os.getenv("DISK_CACHE_ENABLED", "True").lower() == "true"
    DISK_CACHE_PATH: str = os.getenv("DISK_CACHE_PATH", "/tmp/crossmind_disk_cache")
    # DLDB for feedback and knowledge
    DLDB_ENABLED: bool = os.getenv("DLDB_ENABLED", "True").lower() == "true"
    DLDB_PATH: str = os.getenv("DLDB_PATH", "/var/lib/crossmind/dldb")
    # RBAC at application layer
    RBAC_APP_ENABLED: bool = os.getenv("RBAC_APP_ENABLED", "True").lower() == "true"
    DLDB_BACKUP_ENABLED: bool = os.getenv("DLDB_BACKUP_ENABLED", "False").lower() == "true"
    DLDB_MAX_SIZE_GB: float = float(os.getenv("DLDB_MAX_SIZE_GB", "10"))
    DLDB_RETENTION_DAYS: int = int(os.getenv("DLDB_RETENTION_DAYS", "90"))

    # ========== Continuous Ingestion Settings ==========
    CONTINUOUS_INGESTION_INTERVAL: int = int(os.getenv("CONTINUOUS_INGESTION_INTERVAL", "300"))
    INGESTION_BATCH_SIZE: int = int(os.getenv("INGESTION_BATCH_SIZE", "50"))
    INGESTION_MAX_RETRIES: int = int(os.getenv("INGESTION_MAX_RETRIES", "3"))

    # ========== Active Learning Settings ==========
    ACTIVE_LEARNING_ENABLED: bool = os.getenv("ACTIVE_LEARNING_ENABLED", "True").lower() == "true"
    ACTIVE_LEARNING_MIN_FEEDBACK: int = int(os.getenv("ACTIVE_LEARNING_MIN_FEEDBACK", "10"))
    ACTIVE_LEARNING_RETRAIN_INTERVAL: int = int(os.getenv("ACTIVE_LEARNING_RETRAIN_INTERVAL", "3600"))

    # ========== Ingestion Cache Settings ==========
    INGESTION_CACHE_TTL_SECONDS: int = int(os.getenv("INGESTION_CACHE_TTL_SECONDS", "3600"))
    INGESTION_CACHE_MAX_ITEMS: int = int(os.getenv("INGESTION_CACHE_MAX_ITEMS", "10000"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

    # ========== Advanced Reasoning Settings ==========
    MULTI_AGENT_ENABLED: bool = os.getenv("MULTI_AGENT_ENABLED", "True").lower() == "true"
    Z3_VALIDATION_ENABLED: bool = os.getenv("Z3_VALIDATION_ENABLED", "True").lower() == "true"
    DUAL_MEMORY_ENABLED: bool = os.getenv("DUAL_MEMORY_ENABLED", "True").lower() == "true"
    EXPERIMENTAL_BLUEPRINT_ENABLED: bool = os.getenv("EXPERIMENTAL_BLUEPRINT_ENABLED", "True").lower() == "true"
    RISK_FEEDBACK_ENABLED: bool = os.getenv("RISK_FEEDBACK_ENABLED", "True").lower() == "true"

    # ========== Dynamic Ingestion Settings ==========
    DYNAMIC_CONNECTORS_ENABLED: bool = os.getenv("DYNAMIC_CONNECTORS_ENABLED", "True").lower() == "true"
    DYNAMIC_CONNECTOR_SOURCES: str = os.getenv("DYNAMIC_CONNECTOR_SOURCES", "file,api,webhook")
    AUTO_INIT_ON_STARTUP: bool = os.getenv("AUTO_INIT_ON_STARTUP", "True").lower() == "true"

    class Config:
        env_file = ".env"

settings = Settings()