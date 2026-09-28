import asyncio
import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from prometheus_client import Counter, Histogram

from anomaly_detection.base import AnomalyResult, BaseDetector
from anomaly_detection.detector import CONTEXT_CONFIG, ContextualAnomalyDetector, _input_hash

logger = logging.getLogger("crossmind.anomaly_registry")

DETECTION_COUNT = Counter(
    "crossmind_anomaly_detections_total",
    "Anomaly detector evaluations",
    ["context", "label", "action"],
)
DETECTION_SCORE = Histogram(
    "crossmind_anomaly_score",
    "Anomaly detector score distribution",
    ["context"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)


class AnomalyDetectorRegistry:
    def __init__(self):
        model_dir = Path(
            os.environ.get(
                "CROSSMIND_ANOMALY_MODEL_DIR",
                str(Path(__file__).parent / "models"),
            )
        )
        self._detectors: Dict[str, BaseDetector] = {
            context_name: ContextualAnomalyDetector(context_name, model_dir=model_dir)
            for context_name in CONTEXT_CONFIG
        }
        self._cache: Dict[str, AnomalyResult] = {}
        self._cache_lock = threading.RLock()
        self._cache_limit = 4096
        self._observations: Dict[str, int] = {key: 0 for key in self._detectors}
        self._retrain_every = 25
        self._query_windows: Dict[str, deque] = {}
        self._query_window_seconds = 60
        self._query_window_limit = 30

    def register(self, context_name: str, detector: BaseDetector) -> None:
        if not context_name:
            raise ValueError("Detector context name cannot be empty")
        self._detectors[context_name] = detector
        self._observations.setdefault(context_name, 0)
        with self._cache_lock:
            self._cache.clear()

    def contexts(self) -> List[str]:
        return sorted(self._detectors)

    def detector(self, context_name: str) -> BaseDetector:
        try:
            return self._detectors[context_name]
        except KeyError as exc:
            raise ValueError(f"No detector registered for context {context_name!r}") from exc

    async def detect(
        self,
        context_name: str,
        data: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> AnomalyResult:
        detector = self.detector(context_name)
        detection_context = dict(context or {})
        query_window_count = None
        if context_name == "retrieval_query":
            session = str(detection_context.get("session_id", "default"))
            now = time.monotonic()
            with self._cache_lock:
                if session not in self._query_windows and len(self._query_windows) >= self._cache_limit:
                    self._query_windows.pop(next(iter(self._query_windows)))
                window = self._query_windows.setdefault(session, deque())
                while window and now - window[0] > self._query_window_seconds:
                    window.popleft()
                window.append(now)
                query_window_count = len(window)
            detection_context["queries_in_window"] = query_window_count
        model_version = getattr(detector, "_model_version", 0)
        input_hash = _input_hash(data)
        threshold = (context or {}).get("threshold")
        context_key = (
            detection_context.get("session_id"),
            query_window_count,
        ) if context_name == "retrieval_query" else ()
        cache_key = f"{context_name}:{model_version}:{threshold}:{context_key}:{input_hash}"
        with self._cache_lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            result = cached
        else:
            result = await detector.detect(data, detection_context)
            with self._cache_lock:
                if len(self._cache) >= self._cache_limit:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[cache_key] = result

        if query_window_count is not None:
            window_score = min(
                0.99,
                max(0.0, 0.72 + (query_window_count - self._query_window_limit) * 0.01),
            ) if query_window_count > self._query_window_limit else 0.0
            if window_score > result.score:
                result = replace(
                    result,
                    label="anomaly" if window_score >= result.threshold else "normal",
                    score=round(window_score, 6),
                    action=(
                        "force_deep_and_log"
                        if window_score >= result.threshold
                        else "allow"
                    ),
                    details={
                        **result.details,
                        "queries_in_window": query_window_count,
                        "window_seconds": self._query_window_seconds,
                    },
                )
        self._record_observability(result)
        return result

    def detect_sync(
        self,
        context_name: str,
        data: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> AnomalyResult:
        coroutine = self.detect(context_name, data, context)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="anomaly-detection") as executor:
            return executor.submit(asyncio.run, coroutine).result()

    def detect_many_sync(
        self,
        requests: Iterable[Dict[str, Any]],
    ) -> List[AnomalyResult]:
        coroutine = self.detect_many(requests)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="anomaly-detection") as executor:
            return executor.submit(asyncio.run, coroutine).result()

    async def detect_many(
        self,
        requests: Iterable[Dict[str, Any]],
    ) -> List[AnomalyResult]:
        tasks = [
            self.detect(
                request["context"],
                request["data"],
                request.get("metadata"),
            )
            for request in requests
        ]
        return list(await asyncio.gather(*tasks))

    def record_normal_sample(
        self,
        context_name: str,
        data: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        detector = self.detector(context_name)
        add_sample = getattr(detector, "add_normal_sample", None)
        if add_sample is None:
            return
        add_sample(data, context)
        self._observations[context_name] = self._observations.get(context_name, 0) + 1
        if self._observations[context_name] >= self._retrain_every:
            self._observations[context_name] = 0
            retrain = getattr(detector, "retrain_from_observed", None)
            if retrain is not None:
                try:
                    retrain()
                except ValueError as exc:
                    logger.info(
                        "Anomaly model refresh deferred",
                        extra={"anomaly_context": context_name, "reason": str(exc)},
                    )
                with self._cache_lock:
                    self._cache.clear()

    def train(self, context_name: str, known_normal_samples: List[Any]) -> int:
        detector = self.detector(context_name)
        fit = getattr(detector, "fit", None)
        if fit is None:
            raise TypeError(f"Detector {context_name!r} does not support model training")
        model_version = fit(known_normal_samples)
        with self._cache_lock:
            self._cache.clear()
        return model_version

    def _record_observability(self, result: AnomalyResult) -> None:
        DETECTION_COUNT.labels(result.context, result.label, result.action).inc()
        DETECTION_SCORE.labels(result.context).observe(result.score)
        logger.info(
            "anomaly_detection_completed",
            extra={
                "anomaly_context": result.context,
                "anomaly_label": result.label,
                "anomaly_score": result.score,
                "anomaly_confidence": result.confidence,
                "anomaly_action": result.action,
                "anomaly_model": result.model,
                "anomaly_input_hash": result.input_hash,
            },
        )
        from reasoning.dldb import get_dldb

        get_dldb().record_event(
            "anomaly_detection",
            {
                "context": result.context,
                "label": result.label,
                "score": result.score,
                "confidence": result.confidence,
                "threshold": result.threshold,
                "action": result.action,
                "model": result.model,
                "input_hash": result.input_hash,
            },
        )


_registry: Optional[AnomalyDetectorRegistry] = None
_registry_lock = threading.Lock()


def get_anomaly_detector_registry() -> AnomalyDetectorRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = AnomalyDetectorRegistry()
        return _registry


def reset_anomaly_detector_registry() -> None:
    global _registry
    with _registry_lock:
        _registry = None
