from __future__ import annotations

import math
import threading
import time
from collections import deque
from contextlib import AbstractContextManager
from typing import Any, Deque, Dict, Iterable, Optional


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if quantile <= 0:
        return ordered[0]
    if quantile >= 1:
        return ordered[-1]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class LatencyTracker:
    def __init__(self, max_samples: int = 10000):
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        self._samples: Deque[Dict[str, Any]] = deque(maxlen=max_samples)
        self._lock = threading.Lock()
        self._count = 0
        self._error_count = 0
        self._total_ms = 0.0

    def observe(
        self,
        duration_ms: float,
        status: str = "ok",
        operation: str = "request",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")
        sample = {
            "duration_ms": round(float(duration_ms), 6),
            "status": str(status),
            "operation": str(operation),
            "metadata": dict(metadata or {}),
        }
        with self._lock:
            self._samples.append(sample)
            self._count += 1
            self._total_ms += float(duration_ms)
            if status != "ok":
                self._error_count += 1

    def timer(
        self,
        operation: str = "request",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "LatencyTimer":
        return LatencyTimer(self, operation, metadata)

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            samples = list(self._samples)
            total_count = self._count
            total_error_count = self._error_count
            total_ms = self._total_ms
        durations = [sample["duration_ms"] for sample in samples]
        if not durations:
            return {
                "count": 0,
                "total_count": total_count,
                "error_count": 0,
                "total_error_count": total_error_count,
                "error_rate": 0.0,
                "mean_ms": 0.0,
                "total_mean_ms": round(total_ms / total_count, 6) if total_count else 0.0,
                "median_ms": 0.0,
                "p90_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
            }
        window_error_count = sum(1 for sample in samples if sample["status"] != "ok")
        window_total_ms = sum(durations)
        return {
            "count": len(samples),
            "total_count": total_count,
            "error_count": window_error_count,
            "total_error_count": total_error_count,
            "error_rate": round(window_error_count / len(samples), 6),
            "mean_ms": round(window_total_ms / len(samples), 6),
            "total_mean_ms": round(total_ms / total_count, 6) if total_count else 0.0,
            "median_ms": round(percentile(durations, 0.50), 6),
            "p90_ms": round(percentile(durations, 0.90), 6),
            "p95_ms": round(percentile(durations, 0.95), 6),
            "p99_ms": round(percentile(durations, 0.99), 6),
            "min_ms": round(min(durations), 6),
            "max_ms": round(max(durations), 6),
        }


class LatencyTimer(AbstractContextManager[float]):
    def __init__(
        self,
        tracker: LatencyTracker,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self._tracker = tracker
        self._operation = operation
        self._metadata = metadata
        self._started_at = 0.0

    def __enter__(self) -> float:
        self._started_at = time.perf_counter()
        return self._started_at

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        duration_ms = (time.perf_counter() - self._started_at) * 1000.0
        status = "error" if exc_type is not None else "ok"
        metadata = dict(self._metadata or {})
        if exc_type is not None:
            metadata["error_type"] = exc_type.__name__
        self._tracker.observe(duration_ms, status, self._operation, metadata)
        return False


__all__ = ["LatencyTimer", "LatencyTracker", "percentile"]
