import json
import logging
import asyncio
import re
import time
from fastapi import FastAPI, Query, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional

from config import settings
from ingestion.pipeline import get_ingestion_pipeline
from ingestion.seed_data import seed_default_knowledge
from reasoning.neuro_symbolic_pipeline import get_neuro_symbolic_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crossmind.api")

# ========== Security: API Key Auth ==========
security_scheme = HTTPBearer(auto_error=False)

# ========== Security: Rate Limiter (in-memory) ==========
_request_log: Dict[str, list] = {}

def check_rate_limit(client_ip: str) -> bool:
    """Simple in-memory rate limiter: N requests per minute per IP."""
    now = time.time()
    window = 60.0
    if client_ip not in _request_log:
        _request_log[client_ip] = []
    _request_log[client_ip] = [t for t in _request_log[client_ip] if now - t < window]
    if len(_request_log[client_ip]) >= settings.RATE_LIMIT_PER_MINUTE:
        return False
    _request_log[client_ip].append(now)
    return True

def sanitize_input(text: str, max_length: int = None) -> str:
    """Strip dangerous HTML/script content from user input."""
    if not text:
        return ""
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', text, flags=re.IGNORECASE)
    text = re.sub(r'javascript\s*:', '', text, flags=re.IGNORECASE)
    if max_length and len(text) > max_length:
        text = text[:max_length]
    return text.strip()

async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme)):
    """Dependency to verify API key on protected endpoints.
    Only enforces auth if API_KEY environment variable is explicitly set.
    """
    if not settings.API_KEY:
        # No API key configured — allow all requests (dev mode)
        return True
    if credentials is None or credentials.credentials != settings.API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized. Provide a valid API key in the Authorization header (Bearer <key>)."
        )
    return True

app = FastAPI(
    title="CrossMind API",
    description="CrossMind: Neuro-Symbolic AI Scientific Discovery System with Yuuki RxG Nano (1.5B) and Qdrant",
    version=settings.VERSION
)

# Security: Tightened CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# Security: Request size & rate limiting middleware
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.MAX_REQUEST_SIZE_MB * 1024 * 1024:
        return JSONResponse(
            status_code=413,
            content={"detail": f"Request too large. Max size: {settings.MAX_REQUEST_SIZE_MB}MB"}
        )
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again later."}
        )
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

VALID_ROLES = {"public", "researcher", "admin"}

class DocumentIngestRequest(BaseModel):
    documents: List[Dict[str, Any]] = Field(..., description="List of document dicts")

    @validator("documents")
    def validate_documents(cls, docs):
        for doc in docs:
            if "title" in doc:
                doc["title"] = sanitize_input(str(doc["title"]), 500)
            if "content" in doc:
                doc["content"] = sanitize_input(str(doc["content"]), settings.MAX_DOC_CONTENT_LENGTH)
            if "tags" in doc and isinstance(doc["tags"], list):
                doc["tags"] = [sanitize_input(str(t), 100) for t in doc["tags"]]
            if "allowed_roles" in doc and isinstance(doc["allowed_roles"], list):
                doc["allowed_roles"] = [r for r in doc["allowed_roles"] if r in VALID_ROLES]
        return docs

class QueryRequest(BaseModel):
    query: str = Field(..., example="Find cross-domain links between Alzheimer's biomarkers and nanomaterials")
    user_role: str = Field("researcher", example="researcher")

    @validator("query")
    def validate_query(cls, q):
        q = sanitize_input(q, settings.MAX_QUERY_LENGTH)
        if not q:
            raise ValueError("Query cannot be empty")
        return q

    @validator("user_role")
    def validate_role(cls, role):
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role: {role}. Must be one of {VALID_ROLES}")
        return role

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing CrossMind and seeding default scientific knowledge...")
    logger.info(f"API Key Auth: {'Enabled' if settings.API_KEY else 'Disabled (dev mode)'}")
    if not settings.API_KEY:
        logger.info(f"Auto-generated dev API Key: {settings.effective_api_key}")
    try:
        seed_default_knowledge()
        logger.info("Default scientific knowledge seeded successfully.")
    except Exception as e:
        logger.error(f"Error seeding default knowledge: {e}")

@app.get("/")
async def read_root():
    return {
        "project": "CrossMind",
        "engine": "Yuuki RxG Nano (1.5B) + Qdrant Edge",
        "status": "online",
        "version": settings.VERSION,
        "auth": "api_key_required" if settings.API_KEY else "dev_mode_no_auth",
        "endpoints": {
            "ingest": "/api/ingest",
            "query": "/api/query",
            "stream": "/api/stream_reasoning",
            "metrics": "/api/metrics"
        }
    }

@app.post("/api/ingest", dependencies=[Depends(verify_api_key)])
async def ingest_documents(payload: DocumentIngestRequest):
    for doc in payload.documents:
        allowed = doc.get("allowed_roles", [])
        if "admin" not in allowed:
            doc["allowed_roles"] = list(set(allowed) | {"admin"})
    pipeline = get_ingestion_pipeline()
    inserted_ids = pipeline.ingest_documents(payload.documents)
    return {
        "status": "success",
        "ingested_count": len(inserted_ids),
        "inserted_ids": inserted_ids
    }

@app.post("/api/query", dependencies=[Depends(verify_api_key)])
async def execute_query(req: QueryRequest):
    pipeline = get_neuro_symbolic_pipeline()
    result = pipeline.process_query(query=req.query, user_role=req.user_role)
    return result

@app.get("/api/stream_reasoning", dependencies=[Depends(verify_api_key)])
async def stream_reasoning(
    query: str = Query(..., description="Scientific inquiry query"),
    user_role: str = Query("researcher", description="Role for inline RBAC filtering")
):
    query = sanitize_input(query, settings.MAX_QUERY_LENGTH)
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if user_role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid role: {user_role}")

    pipeline = get_neuro_symbolic_pipeline()

    async def sse_generator():
        for event in pipeline.stream_query(query, user_role=user_role):
            yield f"data: {json.dumps(event)}\n\n"
            await asyncio.sleep(0.05)

    return StreamingResponse(sse_generator(), media_type="text/event-stream")

@app.get("/api/metrics")
async def get_metrics():
    return {
        "model": "Yuuki RxG Nano (1.5B)",
        "base_model": "VibeThinker-1.5B (Claude, Gemini, Kimi Distillation)",
        "trainable_params_pct": "1.18% (18.4M parameters)",
        "metrics": {
            "AIME_2024": "80.0%",
            "TruthfulQA_MC1": "89.6%",
            "MMLU_Pro": "65.63%",
            "MMLU": "85.4%",
            "Training_Cost": "< $15",
            "Memory_Footprint": "~3-4 GB VRAM / ~1.5GB RAM for 1M vectors",
            "Total_End_To_End_Latency": "7-14s"
        }
    }

