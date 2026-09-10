"""Structured logging, OpenTelemetry tracking, and Prometheus metrics for the API service."""
import contextvars
import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

REQUESTS = Counter("crossmind_http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("crossmind_http_request_duration_seconds", "HTTP request latency", ["method", "path"])
QUERIES = Counter("crossmind_queries_total", "Completed scientific queries", ["decision"])

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

if OPENTELEMETRY_AVAILABLE:
    trace.set_tracer_provider(TracerProvider())
    _tracer = trace.get_tracer("crossmind.api")
else:
    _tracer = None


class MockSpan:
    def __init__(self, name: str):
        self.name = name
        self.start_time = time.time()
        self.attributes: dict = {}

    def set_attribute(self, key: str, value):
        self.attributes[key] = value

    def end(self):
        pass


class MockTracer:
    def start_as_current_span(self, name: str):
        return MockSpan(name)


@contextmanager
def trace_span(name: str, attributes: dict = None):
    if OPENTELEMETRY_AVAILABLE and _tracer is not None:
        with _tracer.start_as_current_span(name) as span:
            if attributes:
                for k, v in attributes.items():
                    span.set_attribute(k, v)
            yield span
    else:
        span = MockSpan(name)
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        yield span


_request_context: contextvars.ContextVar[Optional["RequestContext"]] = contextvars.ContextVar(
    "request_context", default=None
)


@dataclass
class RequestContext:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    query_id: Optional[str] = None
    user_role: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        data = {
            "correlation_id": self.request_id,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "query_id": self.query_id,
            "user_role": self.user_role,
        }
        if self.trace_id:
            data["trace_id"] = self.trace_id
        if self.span_id:
            data["span_id"] = self.span_id
        data.update(self.extra)
        return data


def get_request_context() -> Optional[RequestContext]:
    return _request_context.get()


def set_request_context(ctx: Optional[RequestContext]) -> None:
    _request_context.set(ctx)


@contextmanager
def request_context(
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    query_id: Optional[str] = None,
    user_role: Optional[str] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    **extra,
):
    ctx = RequestContext(
        request_id=request_id or str(uuid.uuid4()),
        session_id=session_id,
        query_id=query_id,
        user_role=user_role,
        trace_id=trace_id,
        span_id=span_id,
        extra=extra,
    )
    token = _request_context.set(ctx)
    try:
        yield ctx
    finally:
        _request_context.reset(token)


class StructuredJsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        ctx = get_request_context()
        if ctx:
            payload.update(ctx.as_dict())

        duration_ms = getattr(record, "duration_ms", None)
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms

        status_code = getattr(record, "status_code", None)
        if status_code is not None:
            payload["status_code"] = status_code

        method = getattr(record, "method", None)
        if method is not None:
            payload["method"] = method

        path = getattr(record, "path", None)
        if path is not None:
            payload["path"] = path

        error_type = getattr(record, "error_type", None)
        if error_type is not None:
            payload["error_type"] = error_type

        exc_text = getattr(record, "exc_text", None)
        if exc_text:
            payload["exception"] = exc_text

        for key in record.__dict__:
            if key not in payload and not key.startswith("_"):
                try:
                    payload[key] = record.__dict__[key]
                except Exception:
                    pass

        return json.dumps(payload, default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredJsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


def record_request(method: str, path: str, status: int, started_at: float) -> None:
    duration = time.perf_counter() - started_at
    REQUESTS.labels(method, path, str(status)).inc()
    LATENCY.labels(method, path).observe(duration)


def prometheus_payload():
    return generate_latest(), CONTENT_TYPE_LATEST


def log_query(
    logger: logging.Logger,
    message: str,
    query_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_role: Optional[str] = None,
    **kwargs,
) -> None:
    ctx = get_request_context()
    qid = query_id or (ctx.query_id if ctx else None)
    sid = session_id or (ctx.session_id if ctx else None)
    role = user_role or (ctx.user_role if ctx else None)
    extra = {
        "query_id": qid,
        "session_id": sid,
        "user_role": role,
    }
    extra.update(kwargs)
    logger.info(message, extra=extra)


def log_error(
    logger: logging.Logger,
    message: str,
    error: Optional[Exception] = None,
    error_type: Optional[str] = None,
    query_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_role: Optional[str] = None,
    **kwargs,
) -> None:
    ctx = get_request_context()
    qid = query_id or (ctx.query_id if ctx else None)
    sid = session_id or (ctx.session_id if ctx else None)
    role = user_role or (ctx.user_role if ctx else None)
    extra = {
        "query_id": qid,
        "session_id": sid,
        "user_role": role,
        "error_type": error_type or type(error).__name__ if error else "UnknownError",
    }
    if error:
        extra["error_message"] = str(error)
    extra.update(kwargs)
    logger.error(message, extra=extra, exc_info=bool(error))


def log_performance(
    logger: logging.Logger,
    phase: str,
    latency_ms: float,
    query_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **metrics,
) -> None:
    ctx = get_request_context()
    extra = {
        "phase": phase,
        "latency_ms": round(latency_ms, 3),
        "query_id": query_id or (ctx.query_id if ctx else None),
        "session_id": session_id or (ctx.session_id if ctx else None),
    }
    extra.update({k: v for k, v in metrics.items() if v is not None})
    logger.info("performance_metric", extra=extra)
