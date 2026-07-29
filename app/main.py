import json
import logging
import asyncio
import re
import time
import uuid
from fastapi import FastAPI, Query, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional

from config import settings
from ingestion.pipeline import get_ingestion_pipeline
from reasoning.neuro_symbolic_pipeline import get_neuro_symbolic_pipeline
from ingestion.dynamic_connectors import get_dynamic_connectors
from ingestion.ingestion_cache import get_ingestion_cache
from ingestion.active_learning import get_active_learning_engine
from reasoning.risk_feedback import get_risk_feedback_engine
from reasoning.query_cache import get_query_cache
from ingestion.queue_manager import get_queue_manager
from reasoning.benchmark_collector import get_benchmark_collector
from reasoning.feedback_collector import get_feedback_collector
from reasoning.rule_engine import get_rule_engine
from reasoning.rule_updater import get_rule_updater
from reasoning.retrainer import get_model_retrainer
from app.observability import configure_logging, record_request, prometheus_payload, QUERIES

configure_logging()
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
    description="CrossMind: Neuro-Symbolic AI Scientific Discovery System with ZAYA1-8B (8.4B MoE, 760M active) and Qdrant",
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
    started_at = time.perf_counter()
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.MAX_REQUEST_SIZE_MB * 1024 * 1024:
        response = JSONResponse(
            status_code=413,
            content={"detail": f"Request too large. Max size: {settings.MAX_REQUEST_SIZE_MB}MB"}
        )
        response.headers["X-Request-ID"] = request_id
        record_request(request.method, request.url.path, 413, started_at)
        return response
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        response = JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again later."}
        )
        response.headers["X-Request-ID"] = request_id
        record_request(request.method, request.url.path, 429, started_at)
        return response
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["X-Request-ID"] = request_id
    record_request(request.method, request.url.path, response.status_code, started_at)
    logger.info("request_complete", extra={"request_id": request_id, "method": request.method, "path": request.url.path, "status_code": response.status_code, "duration_ms": round((time.perf_counter() - started_at) * 1000, 2)})
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
    query: str = Field(..., example="Find cross-domain connections between energy storage and financial markets")
    user_role: str = Field("researcher", example="researcher")
    confidence_proceed_threshold: float = Field(0.75, ge=0.0, le=1.0)
    confidence_investigate_threshold: float = Field(0.50, ge=0.0, le=1.0)
    session_id: str = Field("default", example="default")

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

    @validator("confidence_investigate_threshold")
    def validate_confidence_thresholds(cls, value, values):
        proceed = values.get("confidence_proceed_threshold", 0.75)
        if value > proceed:
            raise ValueError("confidence_investigate_threshold cannot exceed confidence_proceed_threshold")
        return value

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing CrossMind dynamic ingestion system...")
    logger.info(f"API Key Auth: {'Enabled' if settings.API_KEY else 'Disabled (dev mode)'}")
    if not settings.API_KEY:
        logger.info(f"Auto-generated dev API Key: {settings.effective_api_key}")
    try:
        pipeline = get_ingestion_pipeline()
        pipeline.auto_init()
        get_dynamic_connectors()
        get_ingestion_cache()
        get_active_learning_engine()
        logger.info("Startup initialization complete.")
    except Exception as e:
        logger.error(f"Error during startup initialization: {e}")

@app.get("/")
async def read_root():
    return {
        "project": "CrossMind",
        "engine": "ZAYA1-8B (8.4B MoE) + Qdrant Edge",
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
    result = pipeline.process_query(
        query=req.query,
        user_role=req.user_role,
        confidence_thresholds={"proceed": req.confidence_proceed_threshold, "investigate": req.confidence_investigate_threshold},
        session_id=getattr(req, "session_id", "default"),
    )
    QUERIES.labels(result["confidence_calibration"]["decision"]).inc()
    return result

@app.get("/healthz")
async def healthcheck():
    return {"status": "healthy", "service": settings.PROJECT_NAME}

@app.get("/metrics")
async def prometheus_metrics():
    payload, content_type = prometheus_payload()
    return StreamingResponse(iter([payload]), media_type=content_type)

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
            evt_data = event.get("data", {})
            stage = evt_data.get("stage") if isinstance(evt_data, dict) else None
            if stage in ("thinking", "hypothesis_synthesis"):
                await asyncio.sleep(0.01)
            else:
                await asyncio.sleep(0.02)

    return StreamingResponse(sse_generator(), media_type="text/event-stream")

@app.get("/api/metrics")
async def get_metrics():
    return {
        "model": "ZAYA1-8B (8.4B MoE, 760M active)",
        "base_model": "ZAYA1-8B Q4_K_M Quantized",
        "total_params": "8.4B (760M active per token via MoE)",
        "context_window": "131,072 tokens",
        "quantization": "Q4_K_M (~5.5 GB)",
        "license": "Apache 2.0",
        "metrics": {
            "AIME_2026": "89.1%",
            "AIME_2024": "80.0%",
            "TruthfulQA": "89.6%",
            "MMLU_Pro": "65.63%",
            "MMLU": "85.4%",
            "Training_Cost": "< $500/month (cloud)",
            "Memory_Footprint": "~5.5 GB VRAM (quantized) / ~8.5 GB total pipeline RAM",
            "Active_Parameters": "760M (MoE sparse activation)",
            "Total_End_To_End_Latency": "~1-2s"
        }
    }

@app.get("/api/graph/browser")
async def get_graph_browser_data():
    pipeline = get_neuro_symbolic_pipeline()
    kg = pipeline.knowledge_graph
    nodes = []
    edges = []
    node_ids = set()
    for doc_id, payload in kg.documents.items():
        nid = f"doc:{doc_id}"
        if nid not in node_ids:
            nodes.append({
                "id": nid,
                "label": payload.get("title", doc_id),
                "type": "document",
                "domain": payload.get("domain", "general"),
            })
            node_ids.add(nid)
    for entity, doc_ids in kg.entity_documents.items():
        eid = f"entity:{entity}"
        if eid not in node_ids:
            nodes.append({
                "id": eid,
                "label": entity,
                "type": "entity",
            })
            node_ids.add(eid)
        for doc_id in doc_ids:
            edges.append({
                "source": f"doc:{doc_id}",
                "target": eid,
                "relation": "mentions",
            })
    return {
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }

@app.get("/api/memory/session/{session_id}")
async def get_session_memory(session_id: str):
    pipeline = get_neuro_symbolic_pipeline()
    memory = pipeline.dual_memory.get_session_context(session_id)
    return memory

@app.post("/api/feedback/risk")
async def submit_risk_feedback(payload: Dict[str, Any]):
    engine = get_risk_feedback_engine()
    entry = engine.submit_feedback(
        query=payload.get("query", ""),
        doc_id=payload.get("doc_id", ""),
        score=float(payload.get("score", 0.0)),
        user_role=payload.get("user_role", "researcher"),
        evidence_domains=payload.get("evidence_domains", []),
    )
    return {"status": "recorded", "entry": entry}

@app.get("/api/query/cache")
async def get_query_cache_stats():
    qc = get_query_cache()
    logger.info("Query cache stats requested")
    return {"status": "success", "query_cache_size": qc.size()}

@app.get("/api/queue/stats")
async def get_queue_stats():
    qm = get_queue_manager()
    stats = qm.get_stats()
    logger.info(f"Ingestion queue stats requested: {stats}")
    return {"status": "success", "queue_stats": stats}

@app.get("/api/benchmark")
async def get_benchmark_summary():
    bc = get_benchmark_collector()
    summary = bc.get_summary()
    logger.info("Benchmark summary requested")
    return {"status": "success", "benchmark_summary": summary}

@app.get("/api/feedback/stats")
async def get_feedback_stats():
    fc = get_feedback_collector()
    stats = fc.get_stats()
    logger.info("Feedback collector stats requested")
    return {"status": "success", "feedback_stats": stats}

@app.get("/api/rule-engine/status")
async def get_rule_engine_status():
    engine = get_rule_engine()
    log = engine.get_log()
    stats = {
        "rule_count": len(engine.rules),
        "log_entries": len(log),
        "rules": [r.get("rule_id") for r in engine.rules],
    }
    logger.info("Rule engine status requested")
    return {"status": "success", "rule_engine_stats": stats}

@app.get("/api/rule-updater/rules")
async def get_custom_rules():
    ru = get_rule_updater()
    rules = ru.get_rules()
    logger.info(f"Custom rules list requested: {len(rules)} rules")
    return {"status": "success", "rules": rules}

@app.post("/api/rule-updater/rules")
async def add_custom_rule(payload: Dict[str, Any]):
    ru = get_rule_updater()
    success = ru.add_rule(payload)
    logger.info(f"Custom rule {'added' if success else 'update failed'}: {payload.get('rule_id', 'unknown')}")
    return {"status": "success" if success else "updated", "rule_id": payload.get("rule_id")}

@app.delete("/api/rule-updater/rules/{rule_id}")
async def remove_custom_rule(rule_id: str):
    ru = get_rule_updater()
    success = ru.remove_rule(rule_id)
    logger.info(f"Custom rule removal requested: {rule_id} success={success}")
    return {"status": "success" if success else "not_found", "rule_id": rule_id}

@app.get("/api/model/retrain/status")
async def get_model_retrain_status():
    retrainer = get_model_retrainer()
    status = retrainer.get_status()
    logger.info("Model retrainer status requested")
    return {"status": "success", "model_retrainer_status": status}
