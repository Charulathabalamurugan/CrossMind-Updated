"""Structured logging and Prometheus metrics for the API service."""
import json
import logging
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

REQUESTS = Counter("crossmind_http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("crossmind_http_request_duration_seconds", "HTTP request latency", ["method", "path"])
QUERIES = Counter("crossmind_queries_total", "Completed scientific queries", ["decision"])


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
