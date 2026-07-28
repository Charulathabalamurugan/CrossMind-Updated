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
    # API Key authentication for backend endpoints
    # If empty, API key auth is disabled (not recommended for production)
    API_KEY: str = os.getenv("API_KEY", "")
    # Auto-generate a default API key if none set
    _generated_key: str = ""

    @property
    def effective_api_key(self) -> str:
        if self.API_KEY:
            return self.API_KEY
        if not self._generated_key:
            self._generated_key = secrets.token_urlsafe(32)
        return self._generated_key

    # CORS allowed origins (comma-separated in env, parsed to list)
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501,http://localhost:8000,http://127.0.0.1:8000")

    @property
    def allowed_origins_list(self) -> list:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    # Max request size in MB
    MAX_REQUEST_SIZE_MB: int = int(os.getenv("MAX_REQUEST_SIZE_MB", "10"))

    # Max query length
    MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "5000"))

    # Max document content length
    MAX_DOC_CONTENT_LENGTH: int = int(os.getenv("MAX_DOC_CONTENT_LENGTH", "50000"))

    # Rate limiting (requests per minute per IP)
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "1000"))

    # ========== Qdrant settings ==========
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_IN_MEMORY: bool = os.getenv("QDRANT_IN_MEMORY", "True").lower() == "true"
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "crossmind_knowledge")

    # ========== Yuuki RxG Nano Agent settings ==========
    RXG_NANO_MODEL_NAME: str = os.getenv("RXG_NANO_MODEL_NAME", "OpceanAI/Yuuki-RxG-nano")
    RXG_NANO_API_BASE: str = os.getenv("RXG_NANO_API_BASE", "http://localhost:8000/v1")
    RXG_NANO_TEMPERATURE: float = float(os.getenv("RXG_NANO_TEMPERATURE", "0.2"))
    RXG_NANO_MAX_TOKENS: int = int(os.getenv("RXG_NANO_MAX_TOKENS", "4096"))
    USE_LOCAL_SIMULATOR_FALLBACK: bool = os.getenv("USE_LOCAL_SIMULATOR_FALLBACK", "True").lower() == "true"

    # ========== Embedding settings ==========
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "nomic-ai/nomic-embed-text-v1.5")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "64"))

    # ========== Dynamic Ingestion Settings ==========
    DYNAMIC_CONNECTORS_ENABLED: bool = os.getenv("DYNAMIC_CONNECTORS_ENABLED", "True").lower() == "true"
    DYNAMIC_CONNECTOR_SOURCES: str = os.getenv("DYNAMIC_CONNECTOR_SOURCES", "file,api,webhook")
    AUTO_INIT_ON_STARTUP: bool = os.getenv("AUTO_INIT_ON_STARTUP", "True").lower() == "true"

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

    # ========== Phase 1: Retrieval Settings ==========
    BGE_M3_ENABLED: bool = os.getenv("BGE_M3_ENABLED", "True").lower() == "true"
    MINERU_EXTRACTION_ENABLED: bool = os.getenv("MINERU_EXTRACTION_ENABLED", "True").lower() == "true"
    TIKA_FALLBACK_ENABLED: bool = os.getenv("TIKA_FALLBACK_ENABLED", "True").lower() == "true"
    DOMAIN_QDRANT_COLLECTIONS: bool = os.getenv("DOMAIN_QDRANT_COLLECTIONS", "False").lower() == "true"
    REDIS_QUERY_CACHE_TTL: int = int(os.getenv("REDIS_QUERY_CACHE_TTL", "3600"))
    REDIS_QUERY_CACHE_MAX: int = int(os.getenv("REDIS_QUERY_CACHE_MAX", "5000"))
    BM25_EARLY_TERMINATION_THRESHOLD: float = float(os.getenv("BM25_EARLY_TERMINATION_THRESHOLD", "0.95"))
    PRODUCT_QUANTIZATION_ENABLED: bool = os.getenv("PRODUCT_QUANTIZATION_ENABLED", "True").lower() == "true"
    FALLBACK_DOMAIN: str = os.getenv("FALLBACK_DOMAIN", "general")
    INDEX_UPDATE_INTERVAL_HOURS: int = int(os.getenv("INDEX_UPDATE_INTERVAL_HOURS", "4"))
    FULL_REINDEX_INTERVAL_HOURS: int = int(os.getenv("FULL_REINDEX_INTERVAL_HOURS", "168"))

    # ========== Phase 2: Validation Settings ==========
    GLiNER_ENABLED: bool = os.getenv("GLiNER_ENABLED", "True").lower() == "true"
    DATALOG_RULES_ENABLED: bool = os.getenv("DATALOG_RULES_ENABLED", "True").lower() == "true"
    OPA_ENABLED: bool = os.getenv("OPA_ENABLED", "False").lower() == "true"
    ACTIVE_LEARNING_QUEUE_ENABLED: bool = os.getenv("ACTIVE_LEARNING_QUEUE_ENABLED", "True").lower() == "true"
    TREEINTERPRETER_ENABLED: bool = os.getenv("TREEINTERPRETER_ENABLED", "True").lower() == "true"
    PRIMARY_ONTOLOGY: str = os.getenv("PRIMARY_ONTOLOGY", "UMLS")
    ONTOLOGY_FALLBACKS: str = os.getenv("ONTOLOGY_FALLBACKS", "MGI,ChEBI")

    # ========== Phase 3: Reasoning Settings ==========
    NEO4J_ENABLED: bool = os.getenv("NEO4J_ENABLED", "False").lower() == "true"
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")
    WFAA_FAST_PATH_ENABLED: bool = os.getenv("WFAA_FAST_PATH_ENABLED", "True").lower() == "true"
    FAST_PATH_CONFIDENCE_THRESHOLD: float = float(os.getenv("FAST_PATH_CONFIDENCE_THRESHOLD", "0.85"))
    DEEPSEEK_R1_ENABLED: bool = os.getenv("DEEPSEEK_R1_ENABLED", "False").lower() == "true"
    DEEPSEEK_R1_API_BASE: str = os.getenv("DEEPSEEK_R1_API_BASE", "http://localhost:8000")
    DEEPSEEK_R1_MODEL: str = os.getenv("DEEPSEEK_R1_MODEL", "deepseek-r1-distill-qwen-14b")
    DEEPSEEK_R1_QUANTIZATION: str = os.getenv("DEEPSEEK_R1_QUANTIZATION", "4bit")
    vLLM_ENABLED: bool = os.getenv("vLLM_ENABLED", "False").lower() == "true"
    vLLM_BATCH_SIZE: int = int(os.getenv("vLLM_BATCH_SIZE", "8"))
    GRAPH_RAG_DEPTH: int = int(os.getenv("GRAPH_RAG_DEPTH", "3"))
    ABDUCTIVE_TOP_K: int = int(os.getenv("ABDUCTIVE_TOP_K", "5"))

    # ========== Phase 4: Learning Settings ==========
    STREAMING_SSE_ENABLED: bool = os.getenv("STREAMING_SSE_ENABLED", "True").lower() == "true"
    DLDB_ENABLED: bool = os.getenv("DLDB_ENABLED", "True").lower() == "true"
    ACTIVE_LEARNING_EXPERT_QUEUE_ENABLED: bool = os.getenv("ACTIVE_LEARNING_EXPERT_QUEUE_ENABLED", "True").lower() == "true"
    DRIFT_DETECTION_ENABLED: bool = os.getenv("DRIFT_DETECTION_ENABLED", "True").lower() == "true"
    DRIFT_DETECTION_INTERVAL_HOURS: int = int(os.getenv("DRIFT_DETECTION_INTERVAL_HOURS", "2"))
    DRIFT_KS_THRESHOLD: float = float(os.getenv("DRIFT_KS_THRESHOLD", "0.05"))
    DRIFT_ACCURACY_THRESHOLD: float = float(os.getenv("DRIFT_ACCURACY_THRESHOLD", "0.02"))
    RETRAINING_ENABLED: bool = os.getenv("RETRAINING_ENABLED", "True").lower() == "true"
    RETRAINING_VALIDATION_REQUIRED: bool = os.getenv("RETRAINING_VALIDATION_REQUIRED", "True").lower() == "true"
    PROMETHEUS_ENABLED: bool = os.getenv("PROMETHEUS_ENABLED", "True").lower() == "true"
    GRAFANA_ENABLED: bool = os.getenv("GRAFANA_ENABLED", "True").lower() == "true"
    OPENTELEMETRY_ENABLED: bool = os.getenv("OPENTELEMETRY_ENABLED", "True").lower() == "true"
    HOT_CACHE_ENABLED: bool = os.getenv("HOT_CACHE_ENABLED", "True").lower() == "true"
    DISK_CACHE_ENABLED: bool = os.getenv("DISK_CACHE_ENABLED", "True").lower() == "true"
    SLINKY_OPERATOR_ENABLED: bool = os.getenv("SLINKY_OPERATOR_ENABLED", "False").lower() == "true"
    MLFLOW_REGISTRY_ENABLED: bool = os.getenv("MLFLOW_REGISTRY_ENABLED", "True").lower() == "true"
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    POSTGRES_ENABLED: bool = os.getenv("POSTGRES_ENABLED", "True").lower() == "true"
    POSTGRES_URL: str = os.getenv("POSTGRES_URL", "postgresql://crossmind:crossmind@localhost:5432/crossmind")
    S3_ENABLED: bool = os.getenv("S3_ENABLED", "False").lower() == "true"
    S3_BUCKET: str = os.getenv("S3_BUCKET", "crossmind-cold-storage")
    S3_ENDPOINT: str = os.getenv("S3_ENDPOINT", "http://localhost:9000")
    CELERY_ENABLED: bool = os.getenv("CELERY_ENABLED", "True").lower() == "true"
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
    MODEL_REGISTRY_ENABLED: bool = os.getenv("MODEL_REGISTRY_ENABLED", "True").lower() == "true"
    MODEL_REGISTRY_URI: str = os.getenv("MODEL_REGISTRY_URI", "http://localhost:5000/registry")

    class Config:
        env_file = ".env"

settings = Settings()
