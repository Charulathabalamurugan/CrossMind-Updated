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
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

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
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "256"))

    class Config:
        env_file = ".env"

settings = Settings()
