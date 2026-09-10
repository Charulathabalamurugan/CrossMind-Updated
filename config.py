import os
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        validate_default=True,
        populate_by_name=True,
    )

    ENVIRONMENT: str = Field(default="development", pattern="^(development|testing|staging|production)$")
    SETTINGS_PROFILE: str = Field(default="default", min_length=1, max_length=64)

    PROJECT_NAME: str = Field(default="CrossMind", min_length=1)
    VERSION: str = Field(default="1.0.0", min_length=1)
    HOST: str = Field(default="0.0.0.0", min_length=1)
    PORT: int = Field(default=8000, ge=1, le=65535)

    API_KEY: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("API_KEY", "")), repr=False)
    ALLOWED_ORIGINS: str = Field(
        default="http://localhost:8501,http://127.0.0.1:8501,http://localhost:8000,http://127.0.0.1:8000"
    )
    MAX_REQUEST_SIZE_MB: int = Field(default=10, ge=1, le=1024)
    MAX_QUERY_LENGTH: int = Field(default=5000, ge=1, le=100000)
    MAX_DOC_CONTENT_LENGTH: int = Field(default=50000, ge=1, le=10000000)
    RATE_LIMIT_PER_MINUTE: int = Field(default=1000, ge=1, le=1000000)

    ZAYA1_8B_MODEL_NAME: str = Field(default="ZAYA1-8B", min_length=1)
    ZAYA1_8B_API_BASE: str = Field(default="http://localhost:8000/v1", min_length=1)
    ZAYA1_8B_TEMPERATURE: float = Field(default=0.2, ge=0.0, le=2.0)
    ZAYA1_8B_MAX_TOKENS: int = Field(default=131072, ge=1, le=1000000)
    ZAYA1_8B_TOTAL_PARAMS: int = Field(default=8400000000, ge=1)
    ZAYA1_8B_ACTIVE_PARAMS: int = Field(default=760000000, ge=1)
    ZAYA1_8B_QUANTIZATION: str = Field(default="Q4_K_M", min_length=1)
    ZAYA1_8B_MEMORY_FOOTPRINT_GB: float = Field(default=5.5, gt=0.0)
    ZAYA1_8B_CONTEXT_LENGTH: int = Field(default=131072, ge=1, le=1000000)
    ZAYA1_8B_LICENSE: str = Field(default="Apache 2.0", min_length=1)
    ZAYA1_8B_AIME_2026_SCORE: float = Field(default=89.1, ge=0.0, le=100.0)
    ZAYA1_8B_MOE_ARCHITECTURE: bool = True
    ZAYA1_8B_MARKOVIAN_RSA: bool = True
    ZAYA1_8B_COMPRESSED_ATTENTION: bool = True
    ZAYA1_8B_NATIVE_THINK_BLOCKS: bool = True
    ZAYA1_8B_VLLM_ENABLED: bool = False
    ZAYA1_8B_VLLM_BATCH_SIZE: int = Field(default=8, ge=1, le=1024)
    USE_LOCAL_SIMULATOR_FALLBACK: bool = True

    EMBEDDING_MODEL_NAME: str = Field(default="nomic-ai/nomic-embed-text-v1.5", min_length=1)
    EMBEDDING_DIM: int = Field(default=1024, ge=1, le=16384)

    MINERU_EXTRACTION_ENABLED: bool = True
    MINERU_API_BASE: str = Field(default="http://localhost:8002", min_length=1)
    TIKA_FALLBACK_ENABLED: bool = True
    TIKA_SERVER_URL: str = Field(default="http://localhost:9998", min_length=1)
    BGE_M3_ENABLED: bool = True
    BGE_M3_MODEL_NAME: str = Field(default="BAAI/bge-m3", min_length=1)
    BGE_M3_DIM: int = Field(default=1024, ge=1, le=16384)
    BGE_M3_RETRIEVAL_DIM: int = Field(default=256, ge=1, le=16384)
    BGE_M3_MATRYOSHKA_ENABLED: bool = True
    BGE_M3_PRECISION: str = Field(default="fp32", pattern="^(fp16|fp32|int8)$")
    BGE_M3_MAX_LENGTH: int = Field(default=8192, ge=1, le=100000)
    QDRANT_HOST: str = Field(default="localhost", min_length=1)
    QDRANT_PORT: int = Field(default=6333, ge=1, le=65535)
    QDRANT_IN_MEMORY: bool = True
    QDRANT_COLLECTION_NAME: str = Field(default="crossmind_knowledge", min_length=1)
    PRODUCT_QUANTIZATION_ENABLED: bool = True
    DOMAIN_QDRANT_COLLECTIONS: bool = False
    MULTIVECTOR_SEARCH_ENABLED: bool = False
    SPARSE_VECTOR_ENABLED: bool = False
    TENSOR_3D_INDEXING_ENABLED: bool = False
    REDIS_HOST: str = Field(default="localhost", min_length=1)
    REDIS_PORT: int = Field(default=6379, ge=1, le=65535)
    REDIS_PASSWORD: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("REDIS_PASSWORD", "")), repr=False)
    REDIS_QUERY_CACHE_TTL: int = Field(default=3600, ge=0, le=604800)
    REDIS_QUERY_CACHE_MAX: int = Field(default=5000, ge=1, le=10000000)

    BGE_M3_RETRIEVAL_ENABLED: bool = True
    BM25_ENABLED: bool = True
    BM25_K1: float = Field(default=1.2, ge=0.0)
    BM25_B: float = Field(default=0.75, ge=0.0, le=1.0)
    RRF_ENABLED: bool = True
    RRF_K: int = Field(default=60, ge=1)
    COLBERT_RERANKING_ENABLED: bool = False
    COLBERT_MODEL_NAME: str = Field(default="colbert-ir/colbertv2.0", min_length=1)
    COLBERT_TOP_K: int = Field(default=20, ge=1)
    COLBERT_THRESHOLD: float = Field(default=0.7, ge=0.0, le=1.0)
    COLBERT_MULTIVECTOR_ENABLED: bool = False
    COLBERT_MULTIVECTOR_MAX_SIM: str = Field(default="MAX_SIM", min_length=1)
    COLBERT_MULTIVECTOR_M: int = Field(default=0, ge=0)
    CROSS_ENCODER_RERANKING_ENABLED: bool = True
    CROSS_ENCODER_MODEL_NAME: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", min_length=1)
    CROSS_ENCODER_TOP_K: int = Field(default=20, ge=1)
    CROSS_ENCODER_MAX_CANDIDATES: int = Field(default=40, ge=1)
    SEMANTIC_QUERY_CACHE_THRESHOLD: float = Field(default=0.92, ge=0.0, le=1.0)
    SEMANTIC_QUERY_CACHE_DIM: int = Field(default=256, ge=1, le=16384)
    RBAC_ENABLED: bool = True
    RBAC_ROLES: str = Field(default="admin,analyst,viewer", min_length=1)
    REDIS_RETRIEVAL_CACHE_TTL: int = Field(default=1800, ge=0, le=604800)
    REDIS_RETRIEVAL_CACHE_MAX: int = Field(default=10000, ge=1, le=10000000)

    NEO4J_ENABLED: bool = False
    NEO4J_URI: str = Field(default="bolt://localhost:7687", min_length=1)
    NEO4J_USER: str = Field(default="neo4j", min_length=1)
    NEO4J_PASSWORD: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("NEO4J_PASSWORD", "")), repr=False)

    WFA_FAST_PATH_ENABLED: bool = True
    FAST_PATH_CONFIDENCE_THRESHOLD: float = Field(default=0.85, ge=0.0, le=1.0)
    DECISION_TREE_ENABLED: bool = True
    LITELLM_ENABLED: bool = True
    LITELLM_MODEL_NAME: str = Field(default="lite-llm/mini", min_length=1)
    ZAYA1B_ENABLED: bool = True
    ZAYA1B_MODEL_NAME: str = Field(default="zaya-1b", min_length=1)
    GRAPH_RAG_ENABLED: bool = True
    GRAPH_RAG_DEPTH: int = Field(default=3, ge=1, le=10)
    GRAPH_RAG_EARLY_STOP_THRESHOLD: float = Field(default=0.75, ge=0.0, le=1.0)
    GRAPH_RAG_MAX_PATHS: int = Field(default=20, ge=1, le=1000)
    GRAPH_RAG_MIN_EVIDENCE: int = Field(default=3, ge=1, le=1000)
    VLLM_ENABLED: bool = False
    VLLM_BATCH_SIZE: int = Field(default=8, ge=1, le=1024)
    VLLM_MAX_NUM_SEQS: int = Field(default=2, ge=1, le=1024)
    VLLM_GPU_MEMORY_UTILIZATION: float = Field(default=0.9, ge=0.1, le=1.0)
    VLLM_MAX_MODEL_LEN: int = Field(default=131072, ge=1, le=1000000)
    ZAYA1_8B_REASONING_ENABLED: bool = True
    SCALLOP_ENABLED: bool = False
    SCALLOP_PROGRAM: str = ""
    SEMARA_ENABLED: bool = True
    SEMARA_IMPL: str = Field(default="tech-mahindra", min_length=1)
    SEMARA_ONTOLOGY_URL: str = ""
    SEMARA_OPEN_SOURCE_FALLBACK: bool = True
    DEFORESTVIS_ENABLED: bool = False
    DEFORESTVIS_PORT: int = Field(default=8003, ge=1, le=65535)
    ABDUCTIVE_TOP_K: int = Field(default=5, ge=1, le=1000)

    STREAMLIT_DASHBOARD_ENABLED: bool = True
    REACT_UI_ENABLED: bool = False
    STREAMING_SSE_ENABLED: bool = True
    SSE_COMPRESSION_ENABLED: bool = False
    OPENTELEMETRY_ENABLED: bool = True
    OPENTELEMETRY_EXPORTER: str = Field(default="http://localhost:4317", min_length=1)
    PROMETHEUS_ENABLED: bool = True
    PROMETHEUS_PORT: int = Field(default=9090, ge=1, le=65535)
    HOT_CACHE_ENABLED: bool = True
    DISK_CACHE_ENABLED: bool = True
    DISK_CACHE_PATH: str = Field(default="/tmp/crossmind_disk_cache", min_length=1)
    DLDB_ENABLED: bool = True
    DLDB_PATH: str = Field(default="/var/lib/crossmind/dldb", min_length=1)
    RBAC_APP_ENABLED: bool = True
    DLDB_BACKUP_ENABLED: bool = False
    DLDB_MAX_SIZE_GB: float = Field(default=10, gt=0.0)
    DLDB_RETENTION_DAYS: int = Field(default=90, ge=1, le=36500)
    DASHBOARD_AUTH_ENABLED: bool = False
    DASHBOARD_API_KEY: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("DASHBOARD_API_KEY", "")), repr=False)
    DASHBOARD_ALLOWED_ROLES: str = Field(default="admin,researcher,viewer", min_length=1)

    CONTINUOUS_INGESTION_INTERVAL: int = Field(default=300, ge=1, le=86400)
    INGESTION_BATCH_SIZE: int = Field(default=50, ge=1, le=100000)
    INGESTION_MAX_RETRIES: int = Field(default=3, ge=0, le=20)
    ASYNC_INGESTION_ENABLED: bool = False
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/0", min_length=1)
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/1", min_length=1)

    ACTIVE_LEARNING_ENABLED: bool = True
    ACTIVE_LEARNING_MIN_FEEDBACK: int = Field(default=10, ge=1, le=100000)
    ACTIVE_LEARNING_RETRAIN_INTERVAL: int = Field(default=3600, ge=1, le=604800)

    INGESTION_CACHE_TTL_SECONDS: int = Field(default=3600, ge=0, le=604800)
    INGESTION_CACHE_MAX_ITEMS: int = Field(default=10000, ge=1, le=10000000)
    CHUNK_SIZE: int = Field(default=512, ge=1, le=100000)
    CHUNK_OVERLAP: int = Field(default=64, ge=0, le=100000)

    MULTI_AGENT_ENABLED: bool = True
    Z3_VALIDATION_ENABLED: bool = True
    DUAL_MEMORY_ENABLED: bool = True
    EXPERIMENTAL_BLUEPRINT_ENABLED: bool = True
    RISK_FEEDBACK_ENABLED: bool = True

    DYNAMIC_CONNECTORS_ENABLED: bool = True
    DYNAMIC_CONNECTOR_SOURCES: str = Field(default="file,api,webhook", min_length=1)
    AUTO_INIT_ON_STARTUP: bool = True

    PLUGIN_AUTO_DISCOVER_ENABLED: bool = True
    PLUGIN_DIRS: str = Field(default="", min_length=0)
    REQUEST_DEDUPLICATION_ENABLED: bool = True
    REQUEST_DEDUPLICATION_TTL_SECONDS: int = Field(default=30, ge=0, le=600)
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(default=3, ge=1, le=1000)
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = Field(default=30, ge=1, le=3600)
    EXTERNAL_SERVICE_TIMEOUT_SECONDS: float = Field(default=5.0, gt=0.0, le=120.0)
    TOKEN_TRACKING_ENABLED: bool = True
    COST_PER_1K_TOKENS: float = Field(default=0.0, ge=0.0)
    MODEL_FALLBACK_CHAIN: str = Field(default="vllm,simulator", min_length=1)
    OPENAI_API_KEY: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("OPENAI_API_KEY", "")), repr=False)
    OPENAI_MODEL: str = Field(default="gpt-4o-mini", min_length=1)
    OPENAI_API_BASE: str = Field(default="https://api.openai.com/v1", min_length=1)

    CACHE_SCHEMA_VERSION: int = Field(default=1, ge=1)
    DATA_SCHEMA_VERSION: int = Field(default=1, ge=1)
    VECTOR_SCHEMA_VERSION: int = Field(default=1, ge=1)
    KG_SCHEMA_VERSION: int = Field(default=1, ge=1)
    API_SCHEMA_VERSION: int = Field(default=1, ge=1)

    @property
    def effective_api_key(self) -> str:
        value = self.API_KEY.get_secret_value()
        if value:
            return value
        if not getattr(self, "_generated_key", ""):
            import secrets

            self._generated_key = secrets.token_urlsafe(32)
        return self._generated_key

    @property
    def allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def is_testing(self) -> bool:
        return self.ENVIRONMENT == "testing"

    @model_validator(mode="after")
    def validate_related_values(self) -> "Settings":
        if self.BGE_M3_RETRIEVAL_DIM > self.BGE_M3_DIM:
            raise ValueError("BGE_M3_RETRIEVAL_DIM cannot exceed BGE_M3_DIM")
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.SEMANTIC_QUERY_CACHE_DIM > self.EMBEDDING_DIM:
            raise ValueError("SEMANTIC_QUERY_CACHE_DIM cannot exceed EMBEDDING_DIM")
        return self


def _select_env_file() -> str | None:
    environment = os.getenv("ENVIRONMENT", os.getenv("SETTINGS_PROFILE", "development"))
    profile = os.getenv("SETTINGS_PROFILE", "default")
    candidates = [
        Path.cwd() / f".env.{environment}",
        Path.cwd() / f".env.{profile}",
        Path.cwd() / ".env",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def get_settings() -> Settings:
    return Settings(_env_file=_select_env_file())


settings = get_settings()

__all__ = ["Settings", "get_settings", "settings"]
