"""Structured logging, OpenTelemetry tracking, and Prometheus metrics for the API service."""
import json
import logging
import time
from contextlib import contextmanager
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


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {"timestamp": self.formatTime(record), "level": record.levelname, "logger": record.name, "message": record.getMessage()}
        for key in ("request_id", "method", "path", "status_code", "duration_ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


def record_request(method: str, path: str, status: int, started_at: float) -> None:
    duration = time.perf_counter() - started_at
    REQUESTS.labels(method, path, str(status)).inc()
    LATENCY.labels(method, path).observe(duration)


def prometheus_payload():
    return generate_latest(), CONTENT_TYPE_LATEST
