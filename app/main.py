"""CrossMind FastAPI application entrypoint with versioning, auth, and OpenAPI customization."""

import json
import logging
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, HTTPException, Depends, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi

from config import settings
from app.observability import configure_logging, record_request, prometheus_payload
from app.schemas import (
    QueryRequest,
    DocumentIngestRequest,
    LoginRequest,
    RefreshRequest,
    HealthResponse,
    IngestionResponse,
)
from reasoning.neuro_symbolic_pipeline import NeuroSymbolicPipeline, get_neuro_symbolic_pipeline
from ingestion.pipeline import IngestionPipeline
from reasoning.auth_service import AuthService

configure_logging()
logger = logging.getLogger("crossmind.api")

app = FastAPI(
    title="CrossMind API",
    description="Neuro-Symbolic Discovery Engine API",
    version=settings.VERSION,
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        summary=app.summary,
        description=app.description,
        routes=app.routes,
        webhooks=app.webhooks.routes,
        tags=app.openapi_tags,
        servers=app.servers,
        terms_of_service=getattr(app, "terms_of_service", None),
        contact=getattr(app, "contact", None),
        license_info=getattr(app, "license_info", None),
        separate_input_output_schemas=getattr(app, "separate_input_output_schemas", True),
        external_docs=getattr(app, "openapi_external_docs", None),
    )
    openapi_schema["info"]["x-api-version"] = str(settings.API_SCHEMA_VERSION)
    openapi_schema["info"]["x-versions"] = {
        "current": f"v{settings.API_SCHEMA_VERSION}",
        "deprecated": [],
    }
    openapi_schema.setdefault("components", {}).setdefault("securitySchemes", {})
    openapi_schema["components"]["securitySchemes"]["bearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "api-key-or-jwt",
    }
    openapi_schema.setdefault("security", [])
    openapi_schema["security"].append({"bearerAuth": []})
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def api_versioning_middleware(request: Request, call_next):
    api_version = request.headers.get("X-API-Version", "1")
    request.state.api_version = api_version
    response = await call_next(request)
    response.headers["X-API-Version"] = api_version
    return response


_rate_limit_store: Dict[str, List[float]] = defaultdict(list)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if settings.RATE_LIMIT_PER_MINUTE <= 0:
        return await call_next(request)
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - 60.0
    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip] if t > window_start
    ]
    if len(_rate_limit_store[client_ip]) >= settings.RATE_LIMIT_PER_MINUTE:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please try again later."},
        )
    _rate_limit_store[client_ip].append(now)
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def custom_validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    for error in errors:
        if error.get("type") == "string_too_long":
            loc = error.get("loc", [])
            if loc and loc[-1] == "content":
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request entity too large"},
                )
    return JSONResponse(
        status_code=422,
        content={"detail": [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in errors]},
    )


async def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    if not settings.API_KEY.get_secret_value():
        return None
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = auth_header[7:]
    try:
        from reasoning.auth_service import AuthService
        user = AuthService.validate_token(token)
        if user:
            return {k: v for k, v in user.items() if k != "password_hash"}
    except Exception:
        pass
    if token == settings.effective_api_key:
        return {"username": "api_key_user", "roles": ["admin"]}
    raise HTTPException(status_code=401, detail="Unauthorized")


def _sanitize(value: str) -> str:
    value = re.sub(r'<script[^>]*>.*?</script>', '', value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r'javascript:', '', value, flags=re.IGNORECASE)
    value = re.sub(r'on\w+\s*=', '', value, flags=re.IGNORECASE)
    return value.strip()


@app.get("/", tags=["System"])
def root():
    return {
        "project": settings.PROJECT_NAME,
        "engine": "neuro-symbolic",
        "status": "operational",
        "version": settings.VERSION,
        "endpoints": {
            "query": "/api/query",
            "ingest": "/api/ingest",
            "stream": "/api/stream_reasoning",
            "health": "/healthz",
            "metrics": "/metrics",
            "auth": "/auth/login",
            "docs": "/docs",
        },
    }


@app.get("/healthz", response_model=HealthResponse, tags=["System"])
def healthz():
    return HealthResponse(
        status="healthy",
        service="crossmind-api",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        schema_version=settings.API_SCHEMA_VERSION,
    )


@app.get("/metrics", tags=["System"])
def metrics(request: Request, user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    payload, content_type = prometheus_payload()
    return Response(content=payload, media_type=content_type)


@app.post("/auth/login", tags=["Auth"])
def login(body: LoginRequest):
    user = AuthService.authenticate(body.username, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    tokens = AuthService.create_tokens(user)
    return {
        "access_token": tokens["access_token"],
        "token_type": tokens["token_type"],
        "expires_in": tokens["expires_in"],
        "expires_at": tokens["expires_at"],
        "refresh_token": tokens["refresh_token"],
        "refresh_expires_at": tokens["refresh_expires_at"],
        "user_id": tokens["user_id"],
        "username": tokens["username"],
        "roles": tokens["roles"],
    }


@app.post("/auth/refresh", tags=["Auth"])
def refresh(body: RefreshRequest):
    tokens = AuthService.refresh_access_token(body.refresh_token)
    if not tokens:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    return tokens


@app.get("/auth/users", tags=["Auth"])
def list_users(user: Dict[str, Any] = Depends(get_current_user)):
    if "admin" not in user.get("roles", []):
        raise HTTPException(status_code=403, detail="Forbidden")
    return AuthService.list_users()


@app.post("/api/query", tags=["Query"])
async def api_query(request: Request, body: QueryRequest):
    await get_current_user(request)
    start = time.perf_counter()
    try:
        pipeline = get_neuro_symbolic_pipeline()
        sanitized_query = _sanitize(body.query)
        result = pipeline.process_query(
            query=sanitized_query,
            user_role=body.user_role,
            confidence_thresholds={
                "proceed": body.confidence_proceed_threshold,
                "investigate": body.confidence_investigate_threshold,
            },
            session_id=body.session_id,
        )
        result["query"] = sanitized_query
        record_request("POST", "/api/query", 200, start)
        return result
    except Exception as exc:
        record_request("POST", "/api/query", 500, start)
        logger.error("Query processing failed", exc_info=exc)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/ingest", tags=["Ingestion"])
async def api_ingest(request: Request, body: DocumentIngestRequest):
    await get_current_user(request)
    start = time.perf_counter()
    try:
        pipeline = IngestionPipeline()
        documents = []
        for doc in body.documents:
            doc_dict = doc.model_dump()
            doc_dict["title"] = _sanitize(doc_dict.get("title", ""))
            doc_dict["content"] = _sanitize(doc_dict.get("content", ""))
            documents.append(doc_dict)
        inserted_ids = pipeline.ingest_documents(documents)
        record_request("POST", "/api/ingest", 200, start)
        return IngestionResponse(
            status="success",
            ingested_count=len(inserted_ids),
            inserted_ids=inserted_ids,
            schema_version=settings.API_SCHEMA_VERSION,
        ).model_dump()
    except Exception as exc:
        record_request("POST", "/api/ingest", 500, start)
        logger.error("Ingestion failed", exc_info=exc)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/stream_reasoning", tags=["Streaming"])
async def api_stream_reasoning(request: Request, query: str, user_role: str = "researcher"):
    if not query or not query.strip():
        return JSONResponse(status_code=400, content={"detail": "Query cannot be empty"})
    if user_role not in ("public", "researcher", "admin", "analyst", "viewer"):
        return JSONResponse(status_code=400, content={"detail": "Invalid user_role"})
    
    await get_current_user(request)
    sanitized_query = _sanitize(query)
    
    def generate():
        try:
            pipeline = NeuroSymbolicPipeline()
            for event in pipeline.stream_query(sanitized_query, user_role=user_role):
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except Exception as exc:
            logger.error("Stream error", exc_info=exc)
            yield f"data: {json.dumps({'event': 'error', 'data': {'detail': 'Internal server error'}})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/v1/api/query", tags=["Query"])
async def v1_api_query(request: Request, body: QueryRequest):
    return await api_query(request, body)


@app.post("/v1/api/ingest", tags=["Ingestion"])
async def v1_api_ingest(request: Request, body: DocumentIngestRequest):
    return await api_ingest(request, body)


@app.get("/v1/api/stream_reasoning", tags=["Streaming"])
async def v1_api_stream_reasoning(request: Request, query: str, user_role: str = "researcher"):
    return await api_stream_reasoning(request, query, user_role)


__all__ = ["app"]
